import argparse, json, os, sys, time
from pathlib import Path

def _load_yaml_like(path):
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(txt)
    except Exception:
        return {"raw": txt}

def _merge_dict(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return b
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge_dict(out.get(k), v) if isinstance(v, dict) else v
    return out

def load_config(cfg_path):
    cfg = _load_yaml_like(cfg_path)
    inherit = cfg.get("inherit")
    if inherit:
        base = _load_yaml_like(inherit)
        cfg = _merge_dict(base, {k: v for k, v in cfg.items() if k != "inherit"})
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default="")
    ap.add_argument("--outdir", default="outputs/logs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.overrides.strip():
        try:
            ov = json.loads(args.overrides)
            cfg = _merge_dict(cfg, ov)
        except Exception:
            pass

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = outdir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    msg = {
        "status": "stub",
        "message": "Training entrypoint created. Plug in your framework (MMDetection3D/BEVFormer) here.",
        "config": str(Path(args.config)),
        "run_dir": str(run_dir),
    }
    (run_dir / "status.json").write_text(json.dumps(msg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(msg, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
