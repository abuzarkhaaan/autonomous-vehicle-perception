from pathlib import Path
import json

base_dir = Path.cwd()
nb_dir = base_dir / "notebooks"
nb_dir.mkdir(parents=True, exist_ok=True)

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

dataset_exploration = nb([
    md("# Dataset Exploration\n\nQuick sanity checks for multi-camera inputs and basic dataset stats."),
    code("from datasets.nuscenes_dataset import NuScenesDataset\n\nds = NuScenesDataset(root='../data/nuscenes', split='train')\nprint('len:', len(ds))\ns = ds[0]\nprint('keys:', list(s.keys()))\nprint('num_cams:', len(s['images']))\nprint('image shape:', tuple(s['images'][0].shape))\nprint('gt_boxes_3d:', s['gt_boxes_3d'].shape)\nprint('gt_labels:', s['gt_labels'].shape)\n"),
    code("counts = {}\nfor i in range(min(len(ds), 50)):\n    si = ds[i]\n    n = int(si['gt_labels'].numel())\n    counts[n] = counts.get(n, 0) + 1\nprint('box count histogram (first 50):', dict(sorted(counts.items())))\n"),
])

bev_projection_demo = nb([
    md("# BEV Projection Demo\n\nMinimal BEV grid + placeholder box projection into BEV."),
    code("import torch\nfrom utils.bev_utils import bev_grid, boxes3d_to_bev_boxes\n\ng = bev_grid(h=200, w=200)\nprint('grid:', g.shape)\nboxes3d = torch.tensor([\n    [0.0, 0.0, 0.0, 4.2, 1.8, 1.6, 0.0, 0.0, 0.0],\n    [10.0, -5.0, 0.0, 1.0, 0.6, 1.7, 0.0, 0.0, 0.0],\n], dtype=torch.float32)\nbev_boxes = boxes3d_to_bev_boxes(boxes3d)\nprint('bev boxes:', bev_boxes)\n"),
    code("import matplotlib.pyplot as plt\nimport numpy as np\n\nbev = np.zeros((200, 200), dtype=np.float32)\nplt.figure(figsize=(6, 6))\nplt.imshow(bev)\nfor b in bev_boxes:\n    x1, y1, x2, y2 = b.tolist()\nplt.title('BEV canvas (stub)')\nplt.axis('off')\nplt.show()\n"),
])

qualitative_results = nb([
    md("# Qualitative Results\n\nLoad a prediction JSON (from scripts/inference.py) and create a stub BEV artifact."),
    code("from utils.visualization import load_json, save_json, make_stub_bev_artifact\n\npred_path = '../outputs/predictions'\nprint('Set pred_path to a specific file like ../outputs/predictions/pred_YYYYMMDD-HHMMSS.json')\n"),
    code("import glob\nfrom pathlib import Path\n\ncands = sorted(glob.glob('../outputs/predictions/pred_*.json'))\nprint('found:', len(cands))\nif cands:\n    p = cands[-1]\n    obj = load_json(p)\n    artifact = make_stub_bev_artifact(obj.get('predictions', []), meta={'source': p})\n    out = Path('../outputs/visualizations') / (Path(p).stem + '_artifact.json')\n    save_json(out, artifact)\n    print('saved:', out)\n"),
])

files = {
    "dataset_exploration.ipynb": dataset_exploration,
    "bev_projection_demo.ipynb": bev_projection_demo,
    "qualitative_results.ipynb": qualitative_results,
}

for name, notebook in files.items():
    (nb_dir / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
