"""Reproduce the experiments, tables and figures of the paper.

Steps (paths are relative to the repository root):
  1. download open data (UNOSAT shapefile, CrisisMMD splits; images only with --perception)   data/raw/
  2. calibrate the damage CTMC on UNOSAT                                    results/calibration/
  3. [--perception] perception agent on CrisisMMD (CLIP ViT-L/14, Grad-CAM; GPU recommended)
     -> data/processed/crisismmd_features.csv (a precomputed copy is included in the repository)
  4. evidence-reliability evaluation (Table 1)                              results/perception/
  5. trace-driven replay: 30 paired runs, robustness, social weight, sensitivity (Table 2, Fig. 3)
     -> results/replay/   (about 20 min on one CPU core; --quick: 5 runs into results/replay_quick/)
  6. summaries, paired tests and figure data                                results/replay/, figures/data/
  7. inspection-targeting diagnostics                                       results/replay/diagnostics.json
  8. mutation analysis of the DT-oracle tests (Table 3)                     results/mutation/
  9. compile the figures with pdflatex and export SVG/PNG (skipped if pdflatex is missing)   figures/

Usage:  python reproduce.py [--quick] [--perception] [--skip-replay]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
RAW = os.path.join(ROOT, "data", "raw")
SHP = os.path.join(RAW, "unosat", "extracted", "UNOSAT_Mariupol_Livoberezhyny_DA_12May2022.shp")
FEATURES = os.path.join(ROOT, "data", "processed", "crisismmd_features.csv")


def run(step, *args):
    t0 = time.time()
    print(f"\n=== {step}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    print(f"    done in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="5 replay runs into results/replay_quick (smoke test)")
    ap.add_argument("--perception", action="store_true", help="recompute perception features (downloads images)")
    ap.add_argument("--skip-replay", action="store_true", help="reuse the replay results in results/replay")
    a = ap.parse_args()
    replay = os.path.join(ROOT, "results", "replay_quick" if a.quick else "replay")

    run("1. download data", os.path.join(SRC, "download_data.py"), *(["--images"] if a.perception else []))
    run("2. CTMC calibration on UNOSAT", os.path.join(SRC, "estimate_unosat_q.py"), SHP,
        os.path.join(ROOT, "results", "calibration", "unosat_Q.json"))
    if a.perception:
        run("3. perception agent (CrisisMMD)", os.path.join(SRC, "perception_features.py"),
            os.path.join(RAW, "crisismmd"), os.path.join(ROOT, "results", "perception"), FEATURES)
    run("4. evidence reliability (Table 1)", os.path.join(SRC, "reliability_eval.py"), FEATURES,
        os.path.join(ROOT, "results", "perception"))
    if not a.skip_replay:
        run("5. replay (Table 2, Fig. 3)", os.path.join(SRC, "sim_dt.py"), SHP, FEATURES, replay,
            *(["--quick"] if a.quick else []))
    if not a.quick:
        run("6. summaries and figure data", os.path.join(SRC, "summarize.py"), replay, os.path.join(ROOT, "figures", "data"))
        run("6b. Fig. 2 data", os.path.join(SRC, "make_fig2_data.py"), "--shp", SHP, "--replay", replay)
        run("7. diagnostics", os.path.join(SRC, "diagnostics.py"), SHP, FEATURES, replay)
    run("8. mutation analysis (Table 3)", os.path.join(SRC, "mutation_study.py"), SHP, FEATURES,
        os.path.join(ROOT, "results", "mutation"))
    if shutil.which("pdflatex") and not a.quick:
        run("9. figures", os.path.join(SRC, "export_figures.py"))
    elif not a.quick:
        print("\n=== 9. figures skipped: pdflatex not found (compiled figures are included in figures/)")
    print("\nAll steps finished.")


if __name__ == "__main__":
    main()
