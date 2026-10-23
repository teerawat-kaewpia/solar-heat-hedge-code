"""Cross-platform entry point for the complete analysis pipeline.

Usage: python run_all.py path/to/CRH_merged_30min_zone_level.csv --output-dir results
All generated files are written to ``--output-dir``. The code directory and
the supplied input file are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


CODE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hospital PV analysis pipeline.")
    parser.add_argument("input_csv", type=Path, help="Raw CRH 30-min zone-level portal export (CSV).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Directory for generated results (default: ./outputs). It must not already contain files.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands after validation without running them.")
    return parser.parse_args()


def digest(paths: list[Path]) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if output_dir == CODE_DIR or CODE_DIR in output_dir.parents:
        raise ValueError("--output-dir must be outside the code directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    early_steps: list[tuple[str, list[str]]] = [
        ("a1_prep.py", [str(input_csv)]), ("bootdraws.py", []), ("a2_analysis.py", []),
        ("a4_revision.py", []), ("a3_figs.py", []), ("a5_validation.py", []),
        ("a6_scenarios.py", []), ("a7_robustness.py", []), ("a8_finegrid.py", []),
        ("a3_figs.py", []),
    ]
    later_steps = [
        "a9_robustfig.py", "a10_v8.py", "a11_surfaces_fig.py", "a12_calib_kstar.py",
        "a13_framing.py", "a14_blocklength.py", "a15_metering.py", "a16_fig5.py",
        "a17_bootstrap_compare.py", "a18_block_sensitivity.py", "a19_synthesis_fig.py",
        "a20_export_dataset.py", "a21_reported_values.py", "a22_family_targets.py",
        "a23_fig6.py", "a24_graphical_abstract.py",
    ]
    early = [[sys.executable, str(CODE_DIR / script), *extra] for script, extra in early_steps]
    later = [[sys.executable, str(CODE_DIR / script)] for script in later_steps]

    if args.dry_run:
        print(f"input: {input_csv}")
        print(f"output: {output_dir}")
        for command in early + later:
            print(subprocess.list2cmdline(command))
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for command in early:
        print("+", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=output_dir, check=True)

    protected = sorted(output_dir.glob("fig*.png")) + [output_dir / "results_extra.json"]
    if not protected[:-1] or not protected[-1].is_file():
        raise FileNotFoundError("Expected a3 figure outputs are missing before PDF generation")
    before = digest(protected)
    pdfwrap = [sys.executable, str(CODE_DIR / "pdfwrap.py"), str(CODE_DIR / "a3_figs.py")]
    print("+", subprocess.list2cmdline(pdfwrap), flush=True)
    subprocess.run(pdfwrap, cwd=output_dir, check=True)
    if digest(protected) != before:
        raise RuntimeError("pdfwrap.py changed a protected PNG or results_extra.json")

    for command in later:
        print("+", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=output_dir, check=True)


if __name__ == "__main__":
    main()
