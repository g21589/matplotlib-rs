"""Compare the exact boxplot workload against an archived matplotlib-rs wheel.

Both versions run with the current Python and NumPy in fresh, sequential
processes. No runtime monkeypatches or changes to the workload are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import zipfile

import boxplots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=boxplots.positive, default=5)
    parser.add_argument("--min-speedup", type=float, default=10)
    opts = parser.parse_args()
    output = opts.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = output / "results.json"
    if result.exists():
        raise SystemExit(f"{result} already exists; use a new output directory")
    baseline = output / "baseline/python"
    with zipfile.ZipFile(opts.baseline_wheel) as archive:
        archive.extractall(baseline)
    sources = {"before": baseline, "after": boxplots.ROOT / "python"}
    args = boxplots.arguments([])
    args.output = output
    args.rounds = opts.rounds
    payload = {"baseline_wheel": str(opts.baseline_wheel.resolve()),
               "baseline_sha256": hashlib.sha256(opts.baseline_wheel.read_bytes()).hexdigest(),
               "benchmark_sha256": hashlib.sha256(Path(boxplots.__file__).read_bytes()).hexdigest(),
               "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "warmups": [], "runs": []}
    for round_no in range(opts.rounds + 1):
        for variant in (("before", "after") if round_no % 2 == 0 else ("after", "before")):
            args.python_source = sources[variant]
            label = "warmup" if round_no == 0 else f"round-{round_no:02d}"
            row = boxplots.run_one(args, "rs", f"{label}-{variant}")
            row["variant"] = variant
            payload["warmups" if round_no == 0 else "runs"].append(row)
            result.write_text(json.dumps(payload, indent=2) + "\n")
    rows = payload["runs"] + payload["warmups"]
    assert len({r["data_sha256"] for r in rows}) == 1
    assert len({r["rgba_sha256"] for r in rows}) == 1
    assert all(r["box_count"] == 5400 and r["line_count"] == 32400 for r in rows)
    payload["summary"] = {}
    for variant in sources:
        runs = [r for r in payload["runs"] if r["variant"] == variant]
        payload["summary"][variant] = {
            phase: {"median": statistics.median(r["seconds"][phase] for r in runs),
                    "min": min(r["seconds"][phase] for r in runs),
                    "max": max(r["seconds"][phase] for r in runs)}
            for phase in boxplots.PHASES}
        payload["summary"][variant]["peak_rss_mib"] = statistics.median(r["peak_rss_mib"] for r in runs)
        shutil.copyfile(runs[0]["image"], output / f"{variant}.png")
    payload["build_speedup"] = (payload["summary"]["before"]["build"]["median"] /
                                payload["summary"]["after"]["build"]["median"])
    payload["all_rgba_identical"] = True
    result.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("summary", "build_speedup", "all_rgba_identical")}, indent=2))
    if payload["build_speedup"] < opts.min_speedup:
        raise SystemExit(f"Build speedup {payload['build_speedup']:.2f} < {opts.min_speedup}")


if __name__ == "__main__":
    main()
