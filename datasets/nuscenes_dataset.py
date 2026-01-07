import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

class NuScenesDataset(Dataset):
    def __init__(self, root="data/nuscenes", split="train", transform=None, camera_names=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.camera_names = camera_names or [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        self.index = self._build_index()

    def _build_index(self):
        ann_dir = self.root / "annotations"
        candidates = []
        if ann_dir.exists():
            for p in sorted(ann_dir.rglob("*.json")):
                candidates.append(p)
        if not candidates:
            dummy = [{"id": i, "token": f"sample_{i}"} for i in range(100 if self.split == "train" else 20)]
            return dummy
        items = []
        for p in candidates:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, list):
                    items.extend(obj)
                elif isinstance(obj, dict) and "samples" in obj and isinstance(obj["samples"], list):
                    items.extend(obj["samples"])
                else:
                    items.append({"path": str(p), "content": obj})
            except Exception:
                items.append({"path": str(p)})
        return items

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]
        sample = {
            "id": item.get("id", idx) if isinstance(item, dict) else idx,
            "token": item.get("token", f"sample_{idx}") if isinstance(item, dict) else f"sample_{idx}",
            "camera_names": self.camera_names,
            "images": [torch.zeros(3, 450, 800) for _ in self.camera_names],
            "intrinsics": [torch.eye(3) for _ in self.camera_names],
            "extrinsics": [torch.eye(4) for _ in self.camera_names],
            "gt_boxes_3d": torch.zeros(0, 9),
            "gt_labels": torch.zeros(0, dtype=torch.long),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
