import hashlib
import json
from pathlib import Path

import pandas as pd

from supplementary.visual_robustness.common import RETRY_DIR, RUN_DIR, MAX_FRAMES, model_fingerprints, write_json
from supplementary.visual_robustness.run_experiment1 import config_for, run_trial, build_cassini_zero_contour

MAX_ATTEMPTS = 2


def needs_retry(row):
    return row["status"] != "completed" and (
        row["status"] == "interrupted" or "TimeoutError" in str(row.get("failure_reason")))


def resolved_manifest(original):
    selected, history = [], []
    for row in original.to_dict(orient="records"):
        base = Path(row["folder"])
        history.append({**row, "attempt": 0, "source": "original"})
        chosen = {**row, "attempt": 0, "source": "original",
                  "original_status": row["status"], "original_folder": str(base),
                  "original_failure_reason": row.get("failure_reason")}
        attempts = RETRY_DIR / row["method"] / row["condition"] / f"rep_{int(row['repetition']):02d}"
        for path in sorted(attempts.glob("attempt_*/trial_status.json")):
            status = json.loads(path.read_text(encoding="utf-8"))
            candidate = {**row, **status, "folder": str(path.parent),
                         "attempt": int(path.parent.name.split("_")[-1]), "source": "transport_retry"}
            history.append(candidate)

            if chosen["status"] != "completed":
                chosen.update(candidate)
        selected.append(chosen)
    pd.DataFrame(selected).to_csv(RETRY_DIR / "manifest_resolved.csv", index=False)
    pd.DataFrame(history).to_csv(RETRY_DIR / "attempt_history.csv", index=False)
    return selected


def main():
    original = pd.read_csv(RUN_DIR / "manifest.csv")
    protocol_path = RUN_DIR / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["max_frames"] != MAX_FRAMES or protocol["models"] != model_fingerprints():
        raise ValueError("Original frame count/weights changed; refusing to mix experiments")
    if protocol["config"] != config_for("clean", 0, ""):
        raise ValueError("Original controller settings changed")
    RETRY_DIR.mkdir(parents=True, exist_ok=True)
    recovery = {"original_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                "selection": "Original completed runs retained; first completed transport retry otherwise",
                "communication": "atomic six-camera multipart; 60 s timeout; 5 s resend",
                "original_manifest": str(RUN_DIR / "manifest.csv")}
    path = RETRY_DIR / "recovery_protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != recovery:
        raise ValueError("Recovery protocol changed")
    write_json(path, recovery)
    contour = build_cassini_zero_contour()
    selected = resolved_manifest(original)
    planned = sum(needs_retry(r) for r in selected)
    print(f"Transport trials to recover: {planned}", flush=True)
    for row in selected:
        if not needs_retry(row):
            continue
        parent = RETRY_DIR / row["method"] / row["condition"] / f"rep_{int(row['repetition']):02d}"
        existing = [int(p.name.split("_")[-1]) for p in parent.glob("attempt_*") if p.is_dir()]
        first = max(existing, default=0) + 1
        for number in range(first, first + MAX_ATTEMPTS):
            folder = parent / f"attempt_{number:02d}"
            print(f"START {row['method']} / {row['condition']} / rep_{int(row['repetition']):02d} / attempt_{number:02d}", flush=True)
            status, stop = run_trial(row["method"], row["condition"], int(row["repetition"]), folder, contour)
            resolved_manifest(original)
            print(f"FINISH {row['method']} / {row['condition']} / rep_{int(row['repetition']):02d}: {status}", flush=True)
            if status["status"] == "completed":
                break
            if stop:
                return
            if "TimeoutError" not in str(status.get("failure_reason")):
                break
    final = resolved_manifest(original)
    print(f"RECOVERY SUMMARY: {sum(r['status'] == 'completed' for r in final)}/{len(final)} completed", flush=True)


if __name__ == "__main__":
    main()
