"""
evaluate_cranio.py

Classify and visualize one or more cranial meshes using a trained Net5 model.

Usage (CLI):
    # Single / few meshes
    python evaluate_cranio.py \
        --model   ./results/Net5_metopic_21042026/model.pt \
        --config  ./results/Net5_metopic_21042026/config.yaml \
        --meshes  ./data/synth_data_CN/metopic_normal/control/control_inst_002_CN.ply

    # Entire folder
    python evaluate_cranio.py \
        --model      ./results/Net5_metopic_21042026/model.pt \
        --config     ./results/Net5_metopic_21042026/config.yaml \
        --folder     ./data/synth_data_CN/metopic_normal/control \
        --save-dir   ./results/Net5_metopic_21042026/eval_control

    # Skip interactive 3D viewer
    python evaluate_cranio.py ... --no-3d

Usage (API):
    from evaluate_cranio import evaluate_meshes, evaluate_folder
    results = evaluate_meshes(
        mesh_paths  = ['./data/.../control_001.ply'],
        model_path  = './results/Net5_metopic_21042026/model.pt',
        config_path = './results/Net5_metopic_21042026/config.yaml',
    )
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import yaml
import json
import numpy as np
import torch
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from pathlib import Path

from src.model import Net5
from src.mesh_operations import mesh_to_nvd

# Mesh formats supported by PyVista
MESH_EXTENSIONS = {'.ply', '.obj', '.stl', '.vtk', '.vtp'}

CLASS_MAPS = {
    'sagittal': {'control': 0, 'sagittal': 1},
    'metopic':  {'control': 0, 'metopic': 1},
    'both':     {'control': 0, 'metopic': 1, 'sagittal': 2},
    'all':      {'control': 0, 'metopic': 1, 'sagittal': 2, 'coronal': 3},
}

################################################################################
#  HELPERS
################################################################################

def _load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _load_model(model_path: str, input_features: int, n_classes: int) -> tuple:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device     : {device}')
    print(f'[INFO] Model path : {model_path}')
    model = Net5(input_features=input_features, n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


def _meshes_in_folder(folder: str) -> list:
    """Return sorted list of all mesh file paths in a folder."""
    paths = sorted([
        str(Path(folder) / f)
        for f in os.listdir(folder)
        if Path(f).suffix.lower() in MESH_EXTENSIONS
    ])
    if not paths:
        raise ValueError(f'No mesh files found in: {folder}')
    print(f'[INFO] Found {len(paths)} mesh files in {folder}')
    return paths

################################################################################
#  INFERENCE
################################################################################

def _compute_importance(nvd_array: np.ndarray, model: nn.Module, device: torch.device):
    """
    Single forward pass with gradient tracking.
    Returns predicted class index, softmax probabilities,
    per-feature abs-gradient importance scores, and raw logits.
    """
    input_tensor = torch.tensor(nvd_array, dtype=torch.float32, device=device)
    input_tensor.requires_grad_(True)

    output = model(input_tensor.unsqueeze(0))   # [1, n_classes]
    output.max().backward()

    importance = torch.abs(input_tensor.grad).detach().cpu().numpy()
    input_tensor.requires_grad_(False)

    predicted_class = torch.argmax(output, dim=1).item()
    probs = nn.functional.softmax(output, dim=1).squeeze().detach().cpu().numpy()

    return predicted_class, probs, importance, output.detach().cpu().numpy()


def _predict(mesh_path: str, model: nn.Module,
             class_map: dict, device: torch.device) -> dict:
    """Runs inference on a single mesh. Returns a result dict."""
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f'Mesh not found: {mesh_path}')

    print(f'[INFO] Processing : {mesh_path}')
    nvd_array = mesh_to_nvd(mesh_path)

    pred_idx, probs, importance, logits = _compute_importance(nvd_array, model, device)

    inv_map         = {v: k for k, v in class_map.items()}
    predicted_label = inv_map[pred_idx]
    confidence      = float(probs[pred_idx])
    fp_score        = float(logits.flatten()[pred_idx])

    pv_mesh = pv.read(mesh_path)

    # Print per-class breakdown
    print(f'[RESULT] Predicted : {predicted_label}  '
          f'(confidence {confidence:.1%},  FP score {fp_score:.4f})')
    print(f'         {"Class":<12}  {"Probability":>12}  {"Logit":>8}')
    print(f'         {"-"*12}  {"-"*12}  {"-"*8}')
    logits_flat = logits.flatten()
    for cls, prob, logit in zip(class_map.keys(), probs, logits_flat):
        marker = ' <--' if cls == predicted_label else ''
        print(f'         {cls:<12}  {prob*100:>11.1f}%  {logit:>8.2f}{marker}')
    print()

    return {
        'path':            mesh_path,
        'name':            Path(mesh_path).stem,
        'predicted_label': predicted_label,
        'confidence':      confidence,
        'probabilities':   probs,
        'importance':      importance,
        'fp_score':        fp_score,
        'logits':          logits,
        'pv_mesh':         pv_mesh,
    }

################################################################################
#  MAIN API
################################################################################

def evaluate_meshes(
    mesh_paths:     list,
    model_path:     str,
    config_path:    str,
    show_prob_plot: bool = True,
    show_3d:        bool = True,
    save_dir:       str  = None,
) -> list:
    """
    Classify and optionally visualize a list of mesh files.

    Parameters
    ----------
    mesh_paths      : one or more paths to mesh files (.ply, .obj, ...)
    model_path      : path to model.pt
    config_path     : path to config.yaml
    show_prob_plot  : show the probability bar chart
    show_3d         : show the interactive PyVista 3D viewer
    save_dir        : if provided, saves plots, screenshots, CSV, and JSON

    Returns
    -------
    List of result dicts, one per mesh.
    """
    cfg           = _load_config(config_path)
    class_map     = CLASS_MAPS[cfg['model']['deformation']]
    model, device = _load_model(model_path, cfg['model']['input_features'], len(class_map))

    print(f'[INFO] Class map  : {class_map}')
    print()

    results = [_predict(p, model, class_map, device) for p in mesh_paths]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if show_prob_plot or save_dir:
        _plot_probabilities(results, class_map, show=show_prob_plot, save_dir=save_dir)

    if show_3d:
        _visualize_3d(results, class_map, save_dir=save_dir)

    if save_dir:
        _save_raw(results, class_map, save_dir)
        _save_summary(results, class_map, save_dir)

    return results


def evaluate_folder(
    folder:     str,
    model_path: str,
    config_path: str,
    save_dir:   str,
    show_3d:    bool = False,
) -> list:
    """
    Run evaluation on all mesh files in a folder and save results to save_dir.

    Produces:
      - results.csv        : all filenames, predicted labels, probabilities, logits
      - probabilities.json : full vectors per mesh
      - One probability bar chart per mesh (in save_dir/plots/)

    Parameters
    ----------
    folder      : directory containing mesh files
    model_path  : path to model.pt
    config_path : path to config.yaml
    save_dir    : directory to write all output files
    show_3d     : show interactive 3D viewer per mesh (False by default for batch runs)
    """
    mesh_paths = _meshes_in_folder(folder)
    return evaluate_meshes(
        mesh_paths     = mesh_paths,
        model_path     = model_path,
        config_path    = config_path,
        show_prob_plot = False,   # too many windows for batch; saved to disk instead
        show_3d        = show_3d,
        save_dir       = save_dir,
    )

################################################################################
#  VISUALISATION
################################################################################

def _plot_probabilities(results: list, class_map: dict,
                        show: bool = True, save_dir: str = None):
    """
    One subplot per mesh: bar chart of class probabilities annotated with
    predicted label, confidence, and FP score. Saved as PNG if save_dir is set.
    """
    class_labels = list(class_map.keys())
    n_meshes     = len(results)
    palette      = sns.color_palette('muted', len(class_labels))

    fig, axes = plt.subplots(1, n_meshes, figsize=(4 * n_meshes, 4), sharey=True)
    if n_meshes == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        rows = [{'Class': cls, 'Probability': float(prob)}
                for cls, prob in zip(class_labels, result['probabilities'])]
        df = pd.DataFrame(rows)

        bars = sns.barplot(data=df, x='Class', y='Probability',
                           palette=palette, ax=ax)

        # Annotate each bar with its probability (1 decimal)
        for i, patch in enumerate(bars.patches):
            prob_val = float(result['probabilities'][i])
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 0.02,
                f'{prob_val*100:.1f}%',
                ha='center', va='bottom', fontsize=8
            )

        # Highlight the predicted bar
        pred_idx = class_map[result['predicted_label']]
        bars.patches[pred_idx].set_edgecolor('black')
        bars.patches[pred_idx].set_linewidth(2)

        ax.set_title(
            f"{result['name']}\n"
            f"Pred: {result['predicted_label']}  ({result['confidence']:.1%})\n"
            f"FP score: {result['fp_score']:.3f}",
            fontsize=9
        )
        ax.set_ylim(0, 1.15)
        ax.set_xlabel('')
        if ax is not axes[0]:
            ax.set_ylabel('')

    fig.suptitle('Predicted Class Probabilities', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir:
        out = os.path.join(save_dir, 'probabilities.png')
        plt.savefig(out, bbox_inches='tight', dpi=150)
        print(f'[INFO] Saved probability plot  → {out}')

    if show:
        plt.show()
    else:
        plt.close()


def _visualize_3d(results: list, class_map: dict, save_dir: str = None):
    """
    One PyVista subplot per mesh, colored by gradient-based importance.
    Class probabilities and logits are rendered as text annotations in the viewport.
    Saved as PNG screenshot if save_dir is set.
    """
    n     = len(results)
    shape = (1, n) if n <= 4 else (2, (n + 1) // 2)
    p     = pv.Plotter(shape=shape, border=False,
                       title='Feature Prominence — Gradient Importance')

    for idx, result in enumerate(results):
        row, col = divmod(idx, shape[1])
        p.subplot(row, col)

        mesh = result['pv_mesh']
        imp  = result['importance']

        if len(imp) == mesh.n_points:
            mesh['importance'] = imp
            p.add_mesh(mesh, scalars='importance', cmap='coolwarm',
                       show_scalar_bar=(idx == 0))
        else:
            p.add_mesh(mesh, color='lightslategrey')

        # Build annotation: header + per-class prob and logit
        logits_flat = result['logits'].flatten()
        prob_lines  = '\n'.join(
            f'  {cls:<10} {prob*100:>5.1f}%  logit {logit:>6.2f}'
            for cls, prob, logit in zip(
                class_map.keys(), result['probabilities'], logits_flat)
        )
        annotation = (
            f"{result['name']}\n"
            f"Pred: {result['predicted_label']} ({result['confidence']:.1%})\n"
            f"FP:   {result['fp_score']:.3f}\n"
            f"\n{prob_lines}"
        )
        p.add_text(annotation, font_size=7, position='upper_left')
        p.camera_position = 'xy'

    if save_dir:
        out = os.path.join(save_dir, 'meshes_importance.png')
        p.show(screenshot=out, auto_close=False)
        print(f'[INFO] Saved 3D screenshot     → {out}')
        p.close()
    else:
        p.show()

################################################################################
#  SAVE OUTPUT
################################################################################

def _save_raw(results: list, class_map: dict, save_dir: str):
    """
    Saves probabilities.json with full softmax vectors and logits (1 decimal).
    """
    class_labels = list(class_map.keys())
    json_out     = []

    for r in results:
        json_out.append({
            'name':   r['name'],
            'path':   r['path'],
            'predicted_label': r['predicted_label'],
            'probabilities': {
                cls: round(float(p) * 100, 1)
                for cls, p in zip(class_labels, r['probabilities'])
            },
            'logits': {
                cls: round(float(l), 2)
                for cls, l in zip(class_labels, r['logits'].flatten())
            },
        })

    path = os.path.join(save_dir, 'probabilities.json')
    with open(path, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f'[INFO] Saved raw probabilities → {path}')


def _save_summary(results: list, class_map: dict, save_dir: str):
    """
    Saves results.csv with one row per mesh:
    filename, predicted label, confidence, FP score,
    per-class probability (%), and per-class logit — all at 1 decimal.
    """
    class_labels = list(class_map.keys())
    rows         = []

    for r in results:
        row = {
            'name':            r['name'],
            'path':            r['path'],
            'predicted_label': r['predicted_label'],
            'confidence_%':    round(r['confidence'] * 100, 1),
            'fp_score':        round(r['fp_score'], 4),
        }
        for cls, prob in zip(class_labels, r['probabilities']):
            row[f'prob_{cls}_%'] = round(float(prob) * 100, 1)
        for cls, logit in zip(class_labels, r['logits'].flatten()):
            row[f'logit_{cls}'] = round(float(logit), 2)
        rows.append(row)

    path = os.path.join(save_dir, 'results.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'[INFO] Saved results CSV       → {path}')

################################################################################
#  CLI
################################################################################

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Classify and visualize cranial meshes with a trained Net5 model.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model',  required=True,
        help='Path to model.pt')
    parser.add_argument('--config', required=True,
        help='Path to config.yaml')

    # Input: individual meshes OR an entire folder
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--meshes', nargs='+',
        help='One or more mesh file paths')
    group.add_argument('--folder',
        help='Folder containing mesh files (batch mode)')

    parser.add_argument('--save-dir', default=None,
        help='Directory to save all output files')
    parser.add_argument('--no-prob', action='store_true',
        help='Skip probability bar chart')
    parser.add_argument('--no-3d',   action='store_true',
        help='Skip 3D PyVista viewer')
    return parser.parse_args()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        args = _parse_args()

        if args.folder:
            evaluate_folder(
                folder      = args.folder,
                model_path  = args.model,
                config_path = args.config,
                save_dir    = args.save_dir or os.path.join(args.folder, 'eval'),
                show_3d     = not args.no_3d,
            )
        else:
            evaluate_meshes(
                mesh_paths     = args.meshes,
                model_path     = args.model,
                config_path    = args.config,
                show_prob_plot = not args.no_prob,
                show_3d        = not args.no_3d,
                save_dir       = args.save_dir,
            )

    else:
        # ----------------------------------------------------------------
        # Direct run (PyCharm). Edit the variables below and hit Run.
        # ----------------------------------------------------------------
        EXPERIMENT  = 'Net5_metopic_21042026'
        MODEL_PATH  = f'./results/{EXPERIMENT}/model.pt'
        CONFIG_PATH = f'./results/{EXPERIMENT}/config.yaml'

        # Option A: specific meshes
        MESH_PATHS = [
            r'./data/synth_data_CN/metopic_normal/metopic/metopic_inst_050_CN.ply',
            # add more paths here
        ]

        # Option B: entire folder (set FOLDER, leave MESH_PATHS empty)
        FOLDER   = None   # e.g. r'./data/synth_data_CN/metopic_normal/control'

        SAVE_DIR  = None  # e.g. f'./results/{EXPERIMENT}/eval'
        SHOW_PROB = True
        SHOW_3D   = True
        # ----------------------------------------------------------------

        if FOLDER:
            evaluate_folder(
                folder      = FOLDER,
                model_path  = MODEL_PATH,
                config_path = CONFIG_PATH,
                save_dir    = SAVE_DIR or os.path.join(FOLDER, 'eval'),
                show_3d     = SHOW_3D,
            )
        else:
            evaluate_meshes(
                mesh_paths     = MESH_PATHS,
                model_path     = MODEL_PATH,
                config_path    = CONFIG_PATH,
                show_prob_plot = SHOW_PROB,
                show_3d        = SHOW_3D,
                save_dir       = SAVE_DIR,
            )
