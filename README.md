# Feature Prominence Score for Craniosynostosis

This repository contains the code accompanying the paper:

> **Quantifying dysmorphologies of the neurocranium using artificial neural networks**  
> *Journal of Anatomy*, 2024. https://doi.org/10.1111/joa.14061

The paper introduces an explainable AI approach for the objective classification and severity quantification of craniosynostosis from 3D head meshes. Central to the approach is the **Feature Prominence (FP) score**: a novel metric that captures how prominently a given head shape expresses the characteristics associated with a particular craniosynostosis type.

---

## Overview

Craniosynostosis is a condition in which one or more cranial sutures fuse prematurely, leading to abnormal head shapes. Current clinical assessment is largely subjective, resulting in inconsistent outcomes. This repository provides tools to:

1. Convert 3D head meshes into a compact, privacy-preserving shape representation based on the **Normal Vector Distribution (NVD)**.
2. Train a fully-connected neural network to classify head shapes as normocephalic (control), trigonocephalic (metopic synostosis), or scaphocephalic (sagittal synostosis).
3. Compute gradient-based importance scores per mesh vertex to identify which shape regions drive the classification.
4. Compute the **Feature Prominence (FP) score** to quantify the severity of the detected dysmorphology.

Because the NVD captures shape information only (not raw geometry or texture), the approach is invariant to head size and orientation, and preserves patient anonymity.

---

## Repository structure

```
.
├── src/
│   ├── dataset.py          # PyTorch Dataset: loads meshes and extracts NVDs
│   ├── mesh_operations.py  # Mesh I/O, resampling, and NVD extraction
│   ├── model.py            # Neural network architectures (Net3 – Net6)
│   ├── plot.py             # Training loss plotting utilities
│   └── utils.py            # Coordinate conversion helpers
├── train.py                # Training script
├── evaluate_cranio.py      # Inference, gradient importance, and visualisation
├── config.yaml             # All hyperparameters and paths (edit this, not the scripts)
├── data/                   # (not included) place your mesh data here
└── results/
    └── {experiment_name}/  # one self-contained folder per run (auto-created)
        ├── config.yaml         # copy of the config used for this run
        ├── model.pt            # best model checkpoint
        ├── train_files.json    # file lists per data split
        ├── val_files.json
        ├── test_files.json
        ├── loss_plot.png       # updated each epoch
        ├── roc_auc.png         # per-class ROC-AUC on the test set
        ├── experiment.log      # human-readable full training log
        └── experiment.json     # structured log for programmatic access
```

Every training run is fully self-contained: model weights, data splits, config, plots, and logs all live together in one experiment folder. To reproduce any past run, all inputs are recorded inside that folder.

---

## Method

### Normal Vector Distribution (NVD)

Each 3D mesh is converted to a fixed-length feature vector by computing a kernel density estimate (KDE) over the distribution of surface normal vectors (`mesh_operations.py`). This gives a shape descriptor that is:

- **Low-dimensional** relative to the raw mesh.
- **Size- and orientation-invariant** when meshes are consistently aligned.
- **Privacy-preserving**: the original geometry cannot be reconstructed from the NVD.

A von Mises-Fisher variant (spherical KDE over mesh points rather than normals) is also available via `Von_Misses_Fisher=True` in `mesh_to_nvd`.

### Neural network

A fully-connected network (default: `Net5`, five hidden layers: 256 → 128 → 64 → 32 → n_classes) takes the NVD vector as input and outputs class logits. The architecture is lightweight and trains in minutes on a CPU. Additional architectures (`Net3`, `Net4`, `Net6`) are available in `src/model.py`.

### Feature Prominence score

After training, gradient-based attribution assigns an importance score to each input feature (one per point in the NVD). These importance scores are mapped back to the mesh vertices to highlight which surface regions drove the classification. The **FP score** aggregates these importances into a single scalar — the sum of absolute gradients weighted by prediction confidence — that quantifies how prominently the mesh expresses class-specific shape characteristics, enabling objective severity estimation.

---

## Installation

Python 3.9+ is recommended.

```bash
git clone https://github.com/<your-org>/cranio-nvd.git
cd cranio-nvd
pip install -r requirements.txt
```

### PyTorch and CUDA

`requirements.txt` includes `--extra-index-url` for the PyTorch package server, so `pip install -r requirements.txt` handles the CUDA build automatically. The pinned version is `torch==2.5.1+cu118` (CUDA 11.8), matching the original experimental environment.

For a CPU-only setup, replace the three torch lines in `requirements.txt` with:

```
torch==2.5.1
torchaudio==2.5.1
torchvision==0.20.1
```

### pyacvd on Windows

`pyacvd` has no pre-built wheel for Python 3.11 on Windows. If the pip install fails with a Cython build error, install it via conda first:

```bash
conda install -c conda-forge pyacvd
pip install -r requirements.txt
```

### Key dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.5.1+cu118 | Neural network training and inference |
| `pyvista` | 0.37.0 | 3D mesh I/O and visualisation |
| `pyacvd` | latest | Mesh resampling |
| `vtk` | 9.3.1 | Backend for PyVista |
| `scipy` | 1.13.1 | Gaussian KDE for NVD computation |
| `spherical_kde` | 0.1.2 | Spherical KDE (von Mises-Fisher variant) |
| `scikit-learn` | 1.5.1 | ROC-AUC evaluation |
| `seaborn` / `matplotlib` | 0.13.2 / 3.7.0 | Plotting |
| `pyyaml` | latest | Config file parsing |

---

## Data

The experiments in the paper use the publicly available synthetic 3D head model dataset. Place your mesh files in `./data/` following this folder structure, with one subfolder per class matching the class names in `config.yaml`:

```
data/
└── synth_data_CN/
    └── metopic_sagittal_normal/
        ├── control/
        │   ├── control_inst_001.ply
        │   └── ...
        ├── metopic/
        │   ├── metopic_inst_001.ply
        │   └── ...
        └── sagittal/
            ├── sagittal_inst_001.ply
            └── ...
```

Supported mesh formats: `.ply`, `.obj`, and any format readable by PyVista.

---

## Configuration

All settings live in `config.yaml`. Edit this file before training; the scripts read from it automatically and save a copy into the experiment folder for full reproducibility.

```yaml
experiment:
  name: Net5_metopic_21042026       # used as the experiment folder name
  notes: "Baseline Net5, no augmentation, KDE features, metopic+control"

paths:
  data_folder: ./data/synth_data_CN/metopic_normal
  results_dir: ./results

model:
  input_features: 4515   # must match vertex count after resampling
  deformation: metopic   # sagittal | metopic | both | all

training:
  batch_size: 16
  epochs: 75
  patience: 10
  lr: 0.01
  momentum: 0.9
  seed: 42
```

The `deformation` key controls which class map is used:

| Value | Classes |
|---|---|
| `metopic` | control, metopic |
| `sagittal` | control, sagittal |
| `both` | control, metopic, sagittal |
| `all` | control, metopic, sagittal, coronal |

---

## Training

```bash
python train.py
```

The script:
- Extracts NVDs from all meshes in `data_folder`.
- Splits the data 80/10/10 (train/val/test) with a fixed seed for reproducibility.
- Creates a self-contained experiment folder at `./results/{experiment.name}/`.
- Saves the best checkpoint by validation accuracy, with early stopping controlled by `patience`.
- Uses a cosine annealing learning rate schedule.
- Logs every epoch to both the console and `experiment.log`.
- Evaluates on the held-out test set and saves per-class ROC-AUC curves.

The `experiment.log` produced looks like this:

```
========================================================================
EXPERIMENT : Net5_metopic_21042026
Date       : 2026-04-21 13:12:57
Device     : cuda  (NVIDIA RTX 4000 Ada Generation Laptop GPU)
Notes      : Baseline Net5, no augmentation, KDE features, metopic+control
========================================================================

CONFIGURATION
-------------
  Model           : Net5
  Input features  : 4515
  Deformation     : metopic
  Batch size      : 16
  Max epochs      : 75
  Patience        : 10
  Learning rate   : 0.01
  Momentum        : 0.9
  Seed            : 42

DATA SUMMARY
------------
  Data folder  : ./data/synth_data_CN/metopic_normal
  Total samples: 200
    control     : 100  (50.0%)
    metopic     : 100  (50.0%)

  Split (80/10/10):
    Train : 160 samples
    Val   : 20 samples
    Test  : 20 samples

MODEL ARCHITECTURE
------------------
  Net5(
    (fc1): Linear(in_features=4515, out_features=256, bias=True)
    ...
    (fc5): Linear(in_features=32, out_features=2, bias=True)
  )
  Total parameters     : 1,199,394
  Trainable parameters : 1,199,394

TRAINING LOG
------------
  Epoch  Train Loss   Val Loss   Val Acc         LR
  -----  ----------  ---------  --------  ---------
      1      0.6977     0.7219   45.00%    0.01000  <--
      2      0.5812     0.6103   60.00%    0.00998  <--
     ...

TEST RESULTS
------------
  Test loss     : 0.0412
  Test accuracy : 98.00%

  Per-class ROC-AUC:
    control     : 0.9950
    metopic     : 0.9923
```

The same information is saved as `experiment.json` for programmatic access.

---

## Inference and Feature Prominence scoring

### Direct run (PyCharm / IDE)

Edit the variables at the bottom of `evaluate_cranio.py` and run the file directly:

```python
EXPERIMENT  = 'Net5_metopic_21042026'
MODEL_PATH  = f'./results/{EXPERIMENT}/model.pt'
CONFIG_PATH = f'./results/{EXPERIMENT}/config.yaml'

MESH_PATHS  = [
    r'./data/synth_data_CN/metopic_normal/control/control_inst_002_CN.ply',
    # add more paths here
]

SAVE_DIR  = None   # or e.g. f'./results/{EXPERIMENT}/eval'
SHOW_PROB = True
SHOW_3D   = True
```

### Command line

```bash
# Single mesh
python evaluate_cranio.py \
    --model   ./results/Net5_metopic_21042026/model.pt \
    --config  ./results/Net5_metopic_21042026/config.yaml \
    --meshes  ./data/synth_data_CN/metopic_normal/control/control_inst_002_CN.ply

# Multiple meshes, save all output
python evaluate_cranio.py \
    --model    ./results/Net5_metopic_21042026/model.pt \
    --config   ./results/Net5_metopic_21042026/config.yaml \
    --meshes   ./data/.../control_002.ply ./data/.../metopic_005.ply \
    --save-dir ./results/Net5_metopic_21042026/eval

# Headless (no interactive windows, useful on a server)
python evaluate_cranio.py --model ... --config ... --meshes ... --no-3d
```

### Python API

```python
from evaluate_cranio import evaluate_meshes

results = evaluate_meshes(
    mesh_paths  = ['./data/.../control_002.ply'],
    model_path  = './results/Net5_metopic_21042026/model.pt',
    config_path = './results/Net5_metopic_21042026/config.yaml',
    show_3d     = True,
    save_dir    = './results/Net5_metopic_21042026/eval',
)
```

Each result dict contains:

| Key | Description |
|---|---|
| `predicted_label` | Predicted class name |
| `confidence` | Softmax probability of the predicted class |
| `probabilities` | Full softmax vector over all classes |
| `fp_score` | Feature Prominence score (scalar) |
| `importance` | Per-feature abs-gradient importance array |
| `logits` | Raw network output before softmax |
| `pv_mesh` | PyVista mesh object for further processing |

When `save_dir` is set, the script writes `summary.csv`, `probabilities.json`, a probability bar chart, and a 3D screenshot.

---

## Extending to other classes or anatomies

To add coronal synostosis, set `deformation: all` in `config.yaml`. The class map is resolved automatically:

```
all  →  control=0, metopic=1, sagittal=2, coronal=3
```

For a different dataset, it is recommended to define a custom `Net` under `src/model.py` and set `input_features` to match the vertex count of your resampled meshes. The hardcoded `x.view(-1, N)` in each model's `forward` method must match `input_features`.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{abdelalim2024quantifying,
  title   = {Quantifying dysmorphologies of the neurocranium using artificial neural networks},
  journal = {Journal of Anatomy},
  year    = {2024},
  doi     = {10.1111/joa.14061}
}
```

---

## License

See `LICENSE` for details.
