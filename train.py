import os
import json
import yaml
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from collections import Counter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from src.dataset import HeadDeformationDataset
from src.model import Net5
from src.plot import plot_loss

################################################################################
#  CONFIG
################################################################################

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

experiment_name = cfg['experiment']['name']
notes           = cfg['experiment'].get('notes', '')
data_folder     = cfg['paths']['data_folder']
results_dir     = cfg['paths']['results_dir']

input_features  = cfg['model']['input_features']
deformation     = cfg['model']['deformation']

batch_size      = cfg['training']['batch_size']
epochs          = cfg['training']['epochs']
patience        = cfg['training']['patience']
lr              = cfg['training']['lr']
momentum        = cfg['training']['momentum']
seed            = cfg['training']['seed']

CLASS_MAPS = {
    'sagittal': {'control': 0, 'sagittal': 1},
    'metopic':  {'control': 0, 'metopic': 1},
    'both':     {'control': 0, 'metopic': 1, 'sagittal': 2},
    'all':      {'control': 0, 'metopic': 1, 'sagittal': 2, 'coronal': 3},
}
class_map = CLASS_MAPS[deformation]

################################################################################
#  EXPERIMENT FOLDER
################################################################################

exp_dir    = os.path.join(results_dir, f'{experiment_name}')
model_path = os.path.join(exp_dir, 'model.pt')
os.makedirs(exp_dir, exist_ok=True)

# Save a copy of the config into the experiment folder
shutil.copy('config.yaml', os.path.join(exp_dir, 'config.yaml'))

################################################################################
#  LOGGER
################################################################################

class ExperimentLogger:
    """
    Writes a human-readable .log file and a structured .json file
    to the experiment directory, while also printing to stdout.
    """

    def __init__(self, exp_dir: str, experiment_name: str):
        self.exp_dir   = exp_dir
        self.log_path  = os.path.join(exp_dir, 'experiment.log')
        self.json_path = os.path.join(exp_dir, 'experiment.json')
        self.data      = {'experiment': experiment_name, 'epochs': []}
        self._f        = open(self.log_path, 'w', encoding='utf-8')

    def _write(self, line: str = ''):
        print(line)
        self._f.write(line + '\n')
        self._f.flush()

    def section(self, title: str):
        self._write()
        self._write(title)
        self._write('-' * len(title))

    def line(self, text: str = ''):
        self._write(text)

    def divider(self):
        self._write('=' * 72)

    # ------------------------------------------------------------------

    def log_header(self, device: torch.device):
        self.divider()
        self._write(f'EXPERIMENT : {self.data["experiment"]}')
        self._write(f'Date       : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self._write(f'Device     : {device}' + (
            f'  ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))
        if notes:
            self._write(f'Notes      : {notes}')
        self.divider()
        self.data['date']   = datetime.now().isoformat()
        self.data['device'] = str(device)

    def log_config(self, cfg: dict):
        self.section('CONFIGURATION')
        t = cfg['training']
        m = cfg['model']
        self._write(f'  Model           : Net5')
        self._write(f'  Input features  : {m["input_features"]}')
        self._write(f'  Deformation     : {m["deformation"]}')
        self._write(f'  Batch size      : {t["batch_size"]}')
        self._write(f'  Max epochs      : {t["epochs"]}')
        self._write(f'  Patience        : {t["patience"]}')
        self._write(f'  Learning rate   : {t["lr"]}')
        self._write(f'  Momentum        : {t["momentum"]}')
        self._write(f'  Seed            : {t["seed"]}')
        self.data['config'] = cfg

    def log_data(self, dataset, class_map: dict,
                 train_indices, val_indices, test_indices):
        self.section('DATA SUMMARY')
        total = len(dataset)
        # Per-class counts from the full dataset
        all_labels  = [dataset.data[i][1] for i in range(total)]
        label_counts = Counter(all_labels)
        inv_map      = {v: k for k, v in class_map.items()}

        self._write(f'  Data folder  : {data_folder}')
        self._write(f'  Total samples: {total}')
        for idx, name in inv_map.items():
            n = label_counts[idx]
            self._write(f'    {name:<12}: {n}  ({100*n/total:.1f}%)')

        self._write(f'')
        self._write(f'  Split (80/10/10):')
        self._write(f'    Train : {len(train_indices)} samples')
        self._write(f'    Val   : {len(val_indices)} samples')
        self._write(f'    Test  : {len(test_indices)} samples')

        self.data['data'] = {
            'total': total,
            'per_class': {inv_map[k]: v for k, v in label_counts.items()},
            'train': len(train_indices),
            'val':   len(val_indices),
            'test':  len(test_indices),
        }

    def log_model(self, net: nn.Module):
        self.section('MODEL ARCHITECTURE')
        arch_str = str(net)
        for line in arch_str.splitlines():
            self._write(f'  {line}')
        total_params     = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        self._write(f'')
        self._write(f'  Total parameters     : {total_params:,}')
        self._write(f'  Trainable parameters : {trainable_params:,}')
        self.data['model'] = {
            'architecture': arch_str,
            'total_params': total_params,
        }

    def log_epoch_header(self):
        self.section('TRAINING LOG')
        header = f'  {"Epoch":>5}  {"Train Loss":>10}  {"Val Loss":>9}  {"Val Acc":>8}  {"LR":>9}  {"":>4}'
        self._write(header)
        self._write(f'  {"-"*5}  {"-"*10}  {"-"*9}  {"-"*8}  {"-"*9}  {"-"*4}')

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  val_acc: float, lr: float, is_best: bool):
        flag = '  <--' if is_best else ''
        self._write(
            f'  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  '
            f'{val_acc:>7.2f}%  {lr:>9.5f}{flag}'
        )
        self.data['epochs'].append({
            'epoch': epoch, 'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4), 'val_acc': round(val_acc, 2),
            'lr': round(lr, 6), 'best': is_best,
        })

    def log_training_summary(self, best_acc: float, best_epoch: int,
                              total_epochs: int, elapsed: float, early_stopped: bool):
        self.section('TRAINING SUMMARY')
        self._write(f'  Best val accuracy : {best_acc:.2f}%  (epoch {best_epoch})')
        self._write(f'  Epochs trained    : {total_epochs}')
        self._write(f'  Early stopped     : {"Yes" if early_stopped else "No"}')
        self._write(f'  Training time     : {elapsed/60:.1f} min  ({elapsed:.0f} s)')
        self.data['training_summary'] = {
            'best_val_acc': best_acc, 'best_epoch': best_epoch,
            'total_epochs': total_epochs, 'early_stopped': early_stopped,
            'elapsed_seconds': round(elapsed, 1),
        }

    def log_test(self, test_loss: float, test_acc: float,
                 roc_aucs: dict, class_map: dict):
        inv_map = {v: k for k, v in class_map.items()}
        self.section('TEST RESULTS')
        self._write(f'  Test loss     : {test_loss:.4f}')
        self._write(f'  Test accuracy : {test_acc:.2f}%')
        self._write(f'')
        self._write(f'  Per-class ROC-AUC:')
        for i, roc_auc_val in roc_aucs.items():
            self._write(f'    {inv_map[i]:<12}: {roc_auc_val:.4f}')
        self.divider()
        self.data['test'] = {
            'loss': round(test_loss, 4), 'accuracy': round(test_acc, 2),
            'roc_auc': {inv_map[i]: round(v, 4) for i, v in roc_aucs.items()},
        }

    def save_json(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def close(self):
        self.save_json()
        self._f.close()


################################################################################
#  SETUP
################################################################################

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = ExperimentLogger(exp_dir, experiment_name)
logger.log_header(device)
logger.log_config(cfg)

################################################################################
#  DATA
################################################################################

logger.line()
logger.line('Extracting NVDs from meshes...')
dataset = HeadDeformationDataset(data_folder, class_map)

total_size  = len(dataset)
train_size  = int(0.8 * total_size)
val_size    = int(0.1 * total_size)

indices = list(range(total_size))
np.random.shuffle(indices)

train_indices = indices[:train_size]
val_indices   = indices[train_size:train_size + val_size]
test_indices  = indices[train_size + val_size:]

train_dataset = Subset(dataset, train_indices)
val_dataset   = Subset(dataset, val_indices)
test_dataset  = Subset(dataset, test_indices)

train_files = [dataset.data[i][0] for i in train_indices]
val_files   = [dataset.data[i][0] for i in val_indices]
test_files  = [dataset.data[i][0] for i in test_indices]

for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
    with open(os.path.join(exp_dir, f'{split}_files.json'), 'w') as f:
        json.dump(files, f, indent=2)

logger.log_data(dataset, class_map, train_indices, val_indices, test_indices)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

################################################################################
#  MODEL, LOSS, OPTIMIZER, SCHEDULER
################################################################################

net       = Net5(input_features=input_features, n_classes=len(class_map)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

logger.log_model(net)

################################################################################
#  TRAINING LOOP
################################################################################

best_val_acc          = 0
best_epoch            = 0
epochs_since_improvement = 0
train_loss_list       = []
val_loss_list         = []
early_stopped         = False
t_start               = time.time()

logger.log_epoch_header()

for epoch in range(epochs):

    # --- Train ---
    net.train()
    running_loss = 0.0
    pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch+1}/{epochs}',
                leave=False)

    for inputs, labels, _ in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(net(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.update(1)

    pbar.close()
    train_loss_list.append(running_loss / len(train_dataloader))

    # --- Validate ---
    net.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for nvd_array, labels, _ in val_dataloader:
            nvd_array, labels = nvd_array.to(device), labels.to(device)
            outputs   = net(nvd_array)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss_list.append(val_loss / len(val_dataloader))
    val_acc     = 100 * correct / total
    current_lr  = scheduler.get_last_lr()[0]
    is_best     = val_acc > best_val_acc

    logger.log_epoch(epoch + 1, train_loss_list[-1], val_loss_list[-1],
                     val_acc, current_lr, is_best)

    plot_loss(train_loss_list, val_loss_list, epoch,
              modelname=os.path.join(exp_dir, 'loss_plot'))
    scheduler.step()

    if is_best:
        best_val_acc = val_acc
        best_epoch   = epoch + 1
        torch.save(net.state_dict(), model_path)
        epochs_since_improvement = 0
        if val_acc == 100.0:
            logger.line('  Validation accuracy reached 100%. Stopping early.')
            early_stopped = True
            break
    else:
        epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            logger.line(f'  No improvement for {patience} epochs. Stopping.')
            early_stopped = True
            break

elapsed = time.time() - t_start
logger.log_training_summary(best_val_acc, best_epoch,
                             epoch + 1, elapsed, early_stopped)

################################################################################
#  TEST
################################################################################

logger.line()
logger.line('Loading best model for testing...')
net.load_state_dict(torch.load(model_path, weights_only=True))
net.eval()

test_loss, correct, total = 0.0, 0, 0
true_labels, pred_probs   = [], []

with torch.no_grad():
    for nvd_array, labels, _ in test_dataloader:
        nvd_array, labels = nvd_array.to(device), labels.to(device)
        outputs    = net(nvd_array)
        test_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()
        pred_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

true_labels = np.array(true_labels)
pred_probs  = np.array(pred_probs)
n_classes   = len(class_map)

################################################################################
#  ROC-AUC
################################################################################

roc_aucs = {}
colors   = ['blue', 'red', 'green', 'purple']

plt.figure(figsize=(10, 8))
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(true_labels == i, pred_probs[:, i])
    roc_aucs[i] = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'Class {i} (AUC = {roc_aucs[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC AUC curves per class')
plt.legend(loc='lower right')
plt.savefig(os.path.join(exp_dir, 'roc_auc.png'), bbox_inches='tight')
plt.close()

logger.log_test(test_loss / len(test_dataloader),
                100 * correct / total,
                roc_aucs, class_map)
logger.close()

print(f'\nAll results saved to: {exp_dir}')
