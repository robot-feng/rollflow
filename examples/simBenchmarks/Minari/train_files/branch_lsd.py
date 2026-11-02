"""Create a documented LSD branch from a pure-FM warmup checkpoint.

Only configuration metadata changes; model, optimizer and scheduler states are
copied into the new checkpoint. Continue with run.py train --resume-from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def branch_payload(payload, weight, p_k1=None, precision=None):
    if not 0 < weight < float("inf"):
        raise ValueError("LSD weight must be finite and positive")
    arguments, config = payload["arguments"], payload["head_config"]
    if arguments["variant"] != "rolling_lsd":
        raise ValueError("Expected a rolling_lsd checkpoint")
    if payload["step"] != arguments["fm_warmup"]:
        raise ValueError("Branch only at the FM warmup boundary")
    if float(arguments["w_lsd"]) != float(config["w_lsd"]):
        raise ValueError("Checkpoint arguments and head config disagree")
    if p_k1 is not None:
        if not 0 <= p_k1 <= 1:
            raise ValueError("p_k1 must lie in [0, 1]")
        if float(arguments["p_k1"]) != float(config["p_k1"]):
            raise ValueError("Checkpoint p_k1 arguments and head config disagree")
    if precision is not None and precision not in ('fp32', 'bf16'):
        raise ValueError('precision must be fp32 or bf16')
    for key in ("model", "optimizer", "scheduler", "split", "normalization"):
        if key not in payload:
            raise ValueError(f"Checkpoint is missing {key}")
    old = float(arguments["w_lsd"])
    arguments["w_lsd"] = weight
    config["w_lsd"] = weight
    if p_k1 is not None:
        arguments["p_k1"] = p_k1
        config["p_k1"] = p_k1
    if precision is not None:
        arguments['precision'] = precision
    return old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--w-lsd", type=float, required=True)
    parser.add_argument("--p-k1", type=float, default=None,
                        help="Optionally alter the one-time-group probability after warmup")
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default=None,
                        help='Optionally change autocast precision for the distillation branch')
    parser.add_argument("--output", type=Path, required=True, help="New directory for branch checkpoint and provenance")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old_p_k1 = float(payload["arguments"]["p_k1"]) if args.p_k1 is not None else None
    old_precision = payload['arguments'].get('precision') if args.precision is not None else None
    old_weight = branch_payload(payload, args.w_lsd, args.p_k1, args.precision)
    source_manifest = Path(payload["arguments"]["output"]) / "manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Source run manifest required for resume: {source_manifest}")
    source_hash = digest(args.checkpoint)
    args.output.mkdir(parents=True)
    destination = args.output / "branch.pt"
    temporary = args.output / "branch.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(destination)
    provenance = {"source_checkpoint": str(args.checkpoint.resolve()), "source_sha256": source_hash,
                  "branch_checkpoint": str(destination.resolve()), "branch_sha256": digest(destination),
                  "step": payload["step"], "fm_warmup": payload["arguments"]["fm_warmup"],
                  "source_w_lsd": old_weight, "branch_w_lsd": args.w_lsd,
                  "source_p_k1": old_p_k1, "branch_p_k1": args.p_k1,
                  'source_precision': old_precision, 'branch_precision': args.precision,
                  "changed_fields": ["arguments.w_lsd", "head_config.w_lsd"] +
                      ([] if args.p_k1 is None else ["arguments.p_k1", "head_config.p_k1"]) +
                      ([] if args.precision is None else ['arguments.precision']),
                  "note": "Model, AdamW and scheduler state are copied; resume checks source manifest and data."}
    (args.output / "branch_manifest.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
