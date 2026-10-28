"""Measure upstream-only boxplot optimizations, ablations and the Rust control.

Every variant uses the same workload, official Pillow PNG level 6, fresh
sequential processes, one excluded warmup and a balanced rotating run order.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

import boxplots

VARIANTS = {
    "official": ("mpl", "none"),
    "statistics_only": ("mpl", "statistics"),
    "artists_only": ("mpl", "artists"),
    "python_optimized": ("mpl", "all"),
    "rust_0_0_2": ("rs", "none"),
}


def write_results(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=boxplots.positive, default=5)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    opts = parser.parse_args()
    if "official" not in opts.variants or len(set(opts.variants)) != len(opts.variants):
        parser.error("variants must be unique and include official")
    output = opts.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists():
        parser.error("use a fresh output directory")
    args = boxplots.arguments([])
    args.output = output
    args.rounds = opts.rounds
    args.png_compress_level = 6
    sources = [Path(__file__), Path(boxplots.__file__),
               Path(__file__).with_name("official_boxplot_fastpath.py")]
    payload = {
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "variants": opts.variants, "python": sys.version, "platform": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              cwd=boxplots.ROOT, text=True).strip(),
        "source_sha256": {str(p.relative_to(boxplots.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "environment": {"cpu_visible": os.cpu_count(),
                        "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
                        "memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip()},
        "warmups": [], "runs": [],
    }
    for number in range(opts.rounds + 1):
        order = opts.variants[number % len(opts.variants):] + opts.variants[:number % len(opts.variants)]
        for variant in order:
            args.official_optimization = VARIANTS[variant][1]
            label = ("warmup" if number == 0 else f"round-{number:02d}") + f"-{variant}"
            row = boxplots.run_one(args, VARIANTS[variant][0], label)
            row["variant"] = variant
            payload["warmups" if number == 0 else "runs"].append(row)
            write_results(result_path, payload)
            assert row["box_count"] == 5400 and row["line_count"] == 32400
            if variant != "rust_0_0_2":
                assert not row["rust_extension_loaded"]
                assert row["line_paths_ready_after_build"] == 32400
            if variant in ("artists_only", "python_optimized"):
                assert row["optimization_counts"]["fast_groups"] == 30
                assert row["optimization_counts"]["fallback_groups"] == 0
            if variant in ("statistics_only", "python_optimized"):
                assert row["optimization_counts"]["dense_stats"] == 30
            rows = payload["warmups"] + payload["runs"]
            assert len({r["data_sha256"] for r in rows}) == 1
            assert len({r["rgba_sha256"] for r in rows}) == 1
    summary = {}
    for variant in opts.variants:
        rows = [r for r in payload["runs"] if r["variant"] == variant]
        summary[variant] = {
            phase: {"median": statistics.median(r["seconds"][phase] for r in rows),
                    "min": min(r["seconds"][phase] for r in rows),
                    "max": max(r["seconds"][phase] for r in rows)}
            for phase in boxplots.PHASES}
        summary[variant]["peak_rss_mib"] = statistics.median(r["peak_rss_mib"] for r in rows)
        summary[variant]["png_bytes"] = statistics.median(r["png_bytes"] for r in rows)
        shutil.copyfile(rows[0]["image"], output / f"{variant}.png")
    payload["summary"] = summary
    payload["speedups_vs_official"] = {
        variant: {phase: summary["official"][phase]["median"] / summary[variant][phase]["median"]
                  for phase in boxplots.PHASES}
        for variant in opts.variants}
    rows = payload["warmups"] + payload["runs"]
    payload["all_rgba_identical"] = len({r["rgba_sha256"] for r in rows}) == 1
    payload["all_png_files_identical"] = len({hashlib.sha256(Path(r["image"]).read_bytes()).hexdigest()
                                             for r in rows}) == 1
    write_results(result_path, payload)
    with (output / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "round", *boxplots.PHASES, "peak_rss_mib"])
        writer.writeheader()
        for row in payload["runs"]:
            writer.writerow({"variant": row["variant"], "round": row["round"],
                             **row["seconds"], "peak_rss_mib": row["peak_rss_mib"]})
    lines = ["# Official Matplotlib boxplot experiment", "",
             "Five-round medians in seconds; identical data, 5,400 boxes and 32,400 lines.",
             "PNG encoding uses Pillow level 6 for every implementation.", "",
             "| Variant | Build | First draw | Build + draw | Redraw | Save PNG | Peak RSS MiB | Build speedup |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in opts.variants:
        values = [summary[variant][phase]["median"] for phase in boxplots.PHASES if phase != "import"]
        lines.append(f"| {variant} | " + " | ".join(f"{v:.3f}" for v in values)
                     + f" | {summary[variant]['peak_rss_mib']:.1f} | "
                     + f"{payload['speedups_vs_official'][variant]['build']:.2f}x |")
    lines += ["", f"All RGBA hashes identical: {payload['all_rgba_identical']}",
              f"All saved PNG bytes identical: {payload['all_png_files_identical']}"]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary": summary, "speedups": payload["speedups_vs_official"],
                      "all_rgba_identical": payload["all_rgba_identical"],
                      "all_png_files_identical": payload["all_png_files_identical"]}, indent=2))


if __name__ == "__main__":
    main()
