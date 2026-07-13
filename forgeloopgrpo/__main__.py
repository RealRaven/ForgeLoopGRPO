"""CLI entry point for ForgeLoop GRPO training."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from .main import train
from .utils.data_utils import preview
from .config import validate_config


def main():
    parser = argparse.ArgumentParser(description="ForgeLoop GRPO Training")
    parser.add_argument("command", choices=["train", "preview", "validate"], 
                       help="Command to run")
    parser.add_argument("config", help="Path to config.json")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint dir")
    parser.add_argument("--override", action="append", default=[],
                       help="Override config values (e.g., learning_rate=1e-5)")

    args = parser.parse_args()

    overrides = {}
    for ov in args.override:
        if "=" in ov:
            key, value = ov.split("=", 1)
            try:
                value = eval(value)
            except:
                pass
            overrides[key] = value

    if args.command == "train":
        train(args.config, resume_dir=args.resume, overrides=overrides or None)
    elif args.command == "preview":
        preview(args.config)
    elif args.command == "validate":
        ok = validate_config(args.config)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()