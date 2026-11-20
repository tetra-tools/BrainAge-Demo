"""Command line interface: ``brainage predict | info | export | archive``."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from . import __version__
from . import constants as C

log = logging.getLogger("brainage")

NIFTI_SUFFIXES = (".nii", ".nii.gz")
TABLE_FIELDS = ["subject_id", "input_path", "predicted_age", "chronological_age", "brain_age_gap",
                "status", "error", "model_name", "model_version", "total_seconds"]


# ---------------------------------------------------------------- helpers
def _collect_inputs(args) -> Tuple[List[Path], List[Optional[float]], List[Optional[str]]]:
    """Input paths, chronological ages (rounded down to whole years) and subject ids."""
    from .pipeline import floor_age

    paths, ages, ids = [], [], []
    if args.list:
        delimiter = "," if str(args.list).lower().endswith(".csv") else "\t"
        with open(args.list, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if "path" not in (reader.fieldnames or []):
                raise ValueError(f"--list file must have a 'path' column (found {reader.fieldnames}); "
                                 "use tabs in .tsv/.txt files and commas in .csv files")
            base = Path(args.list).parent
            for row in reader:
                p = Path(row["path"])
                paths.append(p if p.is_absolute() else base / p)
                try:
                    ages.append(floor_age(row.get("age")))  # rounded down, as the training targets
                except ValueError as e:
                    raise ValueError(f"{args.list}, line {reader.line_num}: {e}") from None
                ids.append((row.get("subject_id") or "").strip() or None)
    for item in args.inputs:
        p = Path(item)
        if p.is_dir():
            found = sorted(q for q in p.rglob("*") if q.name.endswith(NIFTI_SUFFIXES))
            if not found:
                log.warning("No NIfTI file found in %s", p)
            paths += found
            ages += [None] * len(found)
            ids += [None] * len(found)
        else:
            paths.append(p)
            ages.append(None)
            ids.append(None)
    if not paths:
        raise ValueError("No input image. Give NIfTI files, folders, or --list file.tsv")
    if args.age is not None:
        if len(paths) != 1:
            raise ValueError("--age can only be used with a single input (use --list for several)")
        ages = [floor_age(args.age)]
    return paths, ages, ids


def _row(r) -> dict:
    d = r.to_dict()
    d["total_seconds"] = d["timings"].get("total")
    return {k: d.get(k) for k in TABLE_FIELDS}


def _write_results(results, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".json":
        output.write_text(json.dumps([r.to_dict() for r in results], indent=2) + "\n")
        return
    delim = "," if output.suffix == ".csv" else "\t"
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_FIELDS, delimiter=delim)
        w.writeheader()
        for r in results:
            w.writerow({k: ("" if v is None else v) for k, v in _row(r).items()})


# --------------------------------------------------------------- commands
def cmd_predict(args) -> int:
    from .pipeline import BrainAgePipeline

    paths, ages, ids = _collect_inputs(args)
    providers = ["CPUExecutionProvider"] if args.cpu else args.providers
    pipe = BrainAgePipeline(args.bundle, providers=providers, num_threads=args.threads, verify=not args.no_verify)
    results = pipe.predict_many(
        paths, chronological_ages=ages, subject_ids=ids, fail_fast=args.fail_fast,
        skip_bias_correction=args.skip_bias_correction, skip_registration=args.skip_registration,
        save_preprocessed_dir=args.save_preprocessed)

    if args.output:
        _write_results(results, Path(args.output))
        log.info("Results written to %s", args.output)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for r in results:
            if r.status == "ok":
                gap = f"  gap={r.brain_age_gap:+.2f}" if r.brain_age_gap is not None else ""
                print(f"{r.input_path}\t{r.predicted_age:.2f} years{gap}")
            else:
                print(f"{r.input_path}\tERROR {r.error}")
    return 0 if all(r.status == "ok" for r in results) else 1


def cmd_info(args) -> int:
    from .bundle import ModelBundle

    bundle = ModelBundle.load(args.bundle, verify=True)
    print(bundle.summary())
    print("Checksums: OK")
    return 0


def cmd_export(args) -> int:
    from .export import archive_bundle, export_bundle

    bundle = export_bundle(args.checkpoint, args.landmarks, args.template, args.mask, args.out,
                           version=args.version, name=args.name, opset=args.opset,
                           mask_threshold=args.mask_threshold, notes=args.notes, force=args.force)
    print(bundle.summary())
    if args.archive:
        print(f"Archive: {archive_bundle(bundle.root)}")
    return 0


def cmd_archive(args) -> int:
    from .export import archive_bundle

    print(archive_bundle(args.bundle_dir, args.output))
    return 0


# ----------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="brainage", description="Brain age prediction from raw T1w MRI.")
    p.add_argument("--version", action="version", version=f"brainage {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logs")
    p.add_argument("-q", "--quiet", action="store_true", help="errors only")
    sub = p.add_subparsers(dest="command", required=True)

    bundle_help = (f"model bundle directory (default: ${C.DEFAULT_BUNDLE_ENV} or {C.DEFAULT_BUNDLE_PATH})")

    pr = sub.add_parser("predict", help="predict brain age from T1w NIfTI images",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="examples:\n"
                               "  brainage predict sub-01_T1w.nii.gz --age 63\n"
                               "  brainage predict /data/bids --output results.tsv\n"
                               "  brainage predict --list subjects.tsv --output results.csv  # columns: path[,age][,subject_id]\n")
    pr.add_argument("inputs", nargs="*", help="NIfTI files (.nii/.nii.gz) or folders (searched recursively)")
    pr.add_argument("--list", help="TSV/CSV with a 'path' column and optional 'age' (rounded down to whole "
                                   "years), 'subject_id'")
    pr.add_argument("--age", type=float,
                    help="chronological age (single input) to compute the brain-age gap; rounded down to "
                         "whole years, as the training targets (63.9 -> 63)")
    pr.add_argument("--bundle", help=bundle_help)
    pr.add_argument("-o", "--output", help="write results to .tsv, .csv or .json")
    pr.add_argument("--json", action="store_true", help="print full JSON results on stdout")
    pr.add_argument("--save-preprocessed", metavar="DIR", help="save registered image and model input NIfTI")
    pr.add_argument("--skip-bias-correction", action="store_true", help="input is already N4-corrected")
    pr.add_argument("--skip-registration", action="store_true",
                    help="input is already rigid/affine registered to the bundle template (same grid)")
    pr.add_argument("--threads", type=int,
                    help="CPU threads for ANTs / torch / ONNX Runtime; --threads 1 makes the registration "
                         "bit-reproducible (about 2.5x slower)")
    pr.add_argument("--cpu", action="store_true", help="force CPUExecutionProvider")
    pr.add_argument("--providers", nargs="+", help="explicit ONNX Runtime providers")
    pr.add_argument("--no-verify", action="store_true", help="skip bundle checksum verification")
    pr.add_argument("--fail-fast", action="store_true", help="stop at the first failing image")
    pr.set_defaults(func=cmd_predict)

    pi = sub.add_parser("info", help="show and verify a model bundle")
    pi.add_argument("--bundle", help=bundle_help)
    pi.set_defaults(func=cmd_info)

    pe = sub.add_parser("export", help="convert a Lightning checkpoint into a model bundle")
    pe.add_argument("--checkpoint", required=True, help="SFCN Lightning checkpoint (.ckpt)")
    pe.add_argument("--landmarks", required=True, help="t1_landmarks.npy used to build the training cache")
    pe.add_argument("--template", required=True, help="registration template, e.g. ravg152T1_1mm.nii")
    pe.add_argument("--mask", required=True, help="brain mask source, e.g. ravg152T1_1mm_defaced.nii")
    pe.add_argument("--out", required=True, help="output bundle directory")
    pe.add_argument("--version", required=True, help="model version, e.g. 1.0.0")
    pe.add_argument("--name", default="brainage-sfcn")
    pe.add_argument("--opset", type=int, default=18)
    pe.add_argument("--mask-threshold", type=float, default=C.MASK_THRESHOLD)
    pe.add_argument("--notes", default="", help="free text stored in the manifest (dataset, MAE, …)")
    pe.add_argument("--archive", action="store_true", help="also produce <out>.tar.gz + .sha256")
    pe.add_argument("--force", action="store_true", help="overwrite --out")
    pe.set_defaults(func=cmd_export)

    pa = sub.add_parser("archive", help="pack a bundle directory as .tar.gz + .sha256")
    pa.add_argument("bundle_dir")
    pa.add_argument("-o", "--output")
    pa.set_defaults(func=cmd_archive)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.ERROR if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError) as e:  # OSError: missing file, permission denied, …
        log.error("%s", e)
        if args.verbose:
            raise
        return 2


if __name__ == "__main__":
    sys.exit(main())
