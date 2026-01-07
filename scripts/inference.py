import argparse, json, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", default="")
    ap.add_argument("--outdir", default="outputs/predictions")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    pred_path = outdir / f"pred_{run_id}.json"

    payload = {
        "status": "stub",
        "message": "Inference entrypoint created. Load checkpoint and run forward pass here.",
        "config": str(Path(args.config)),
        "checkpoint": args.checkpoint,
        "input": args.input,
        "predictions": [],
    }
    pred_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
