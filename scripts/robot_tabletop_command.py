"""Allowlisted English text commands for isolated simulated bottle placement.

An explicit --validate OR --execute is mandatory. No listening, free-form
planning, retries, or automatic validation followed by execution is provided.
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_COMMAND = "pick up the bottle and place it in the blue tray"
# Full-string grammar: only these words and harmless spacing/case/punctuation
# are accepted. In particular, negation or an appended second task is rejected.
COMMAND_PATTERN = re.compile(
    r"(?:please )?pick up (?:the )?bottle and (?:place|put) it "
    r"(?:in|into) (?:the )?blue (?:tray|track)[.!]?", re.ASCII
)
# Entirely constant shell program. User text is NEVER passed to the shell, nor
# inserted into argv. Paths and the single validated mode are positional args.
ROS_EXEC = 'source /opt/ros/jazzy/setup.bash && exec "$@"'


def parse_command(command: str) -> str:
    normalized = " ".join(command.lower().split())
    if not COMMAND_PATTERN.fullmatch(normalized):
        raise ValueError(f"Unsupported command. Supported task: {CANONICAL_COMMAND!r}")
    return CANONICAL_COMMAND


def launcher_argv(mode: str) -> list[str]:
    if mode not in ("--validate", "--execute"):
        raise ValueError("Explicit --validate or --execute required")
    return ["/bin/bash", "--noprofile", "--norc", "-c", ROS_EXEC, "robot-tabletop-command",
            str(ROOT / "env_isaacsim/bin/python"), str(ROOT / "isaac/scripts/run_bottle_pick.py"),
            "--yolo-place", "--trained-tabletop", mode]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, help=f"Supported task: {CANONICAL_COMMAND}")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate", action="store_true", help="Run isolated headless simulation validation only")
    modes.add_argument("--execute", action="store_true", help="Run visible simulation; requires matching successful validation")
    args = parser.parse_args(argv)
    try:
        task = parse_command(args.command)
    except ValueError as error:
        parser.error(str(error))
    mode = "--validate" if args.validate else "--execute"
    print(f"Accepted task: {task}; mode: {mode}; simulation only", flush=True)
    try:
        return subprocess.run(launcher_argv(mode), cwd=ROOT, check=False).returncode
    except OSError as error:
        print(f"Could not start launcher: {error}. No retry or automatic release.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())