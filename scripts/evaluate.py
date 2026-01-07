import argparse, json, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--outdir", default="outputs/predictions")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    report_path = outdir / f"eval_{run_id}.json"

    report = {
        "status": "stub",
        "message": "Evaluation entrypoint created. Compute nuScenes mAP/NDS here.",
        "config": str(Path(args.config)),
        "checkpoint": args.checkpoint,
        "metrics": {},
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
