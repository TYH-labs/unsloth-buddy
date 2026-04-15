"""
add_reflect_hint.py — Append or list inline reflection hints in .reflect_hints.json.

Usage:
    # Append a new hint
    python3 add_reflect_hint.py <project_dir> --phase <N> --hint "<text>" [--type lesson|skill|user]

    # List existing hints (check before adding on resume)
    python3 add_reflect_hint.py <project_dir> --list

Appends to <project_dir>/.reflect_hints.json (creates if absent).
Call this immediately after confirming a workaround or non-obvious discovery
during phases 2–6. Phase 7 `reflect.py --extract` picks the file up automatically.

On resume: run --list first to see what is already captured; do not re-add known hints.

Do NOT capture routine parameter choices already recorded in gaslamp.md.
Only capture: silent failures, hardware-specific bugs, version incompatibilities,
unexpected hyperparameter behaviours.
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def main():
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        print(__doc__, file=sys.stderr)
        sys.exit(0)

    list_mode = "--list" in args

    project_dir = None
    phase = None
    hint = None
    type_hint = None

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--list":
            i += 1
            continue
        if not a.startswith("--") and project_dir is None:
            project_dir = Path(a)
        elif a == "--phase":
            i += 1
            if i >= len(args):
                print("Error: --phase requires an integer argument.", file=sys.stderr)
                sys.exit(1)
            try:
                phase = int(args[i])
            except ValueError:
                print(f"Error: --phase must be an integer. Got: {args[i]}", file=sys.stderr)
                sys.exit(1)
        elif a == "--hint":
            i += 1
            if i >= len(args):
                print("Error: --hint requires a text argument.", file=sys.stderr)
                sys.exit(1)
            hint = args[i]
        elif a == "--type":
            i += 1
            if i >= len(args):
                print("Error: --type requires lesson|skill|user.", file=sys.stderr)
                sys.exit(1)
            type_hint = args[i]
            if type_hint not in ("lesson", "skill", "user"):
                print(
                    f"Error: --type must be lesson, skill, or user. Got: {type_hint}",
                    file=sys.stderr,
                )
                sys.exit(1)
        i += 1

    if not project_dir:
        print("Error: project_dir argument is required.", file=sys.stderr)
        sys.exit(1)
    if not project_dir.is_dir():
        print(f"Error: not a directory: {project_dir}", file=sys.stderr)
        sys.exit(1)

    hints_file = project_dir / ".reflect_hints.json"

    # ── List mode ──────────────────────────────────────────────────────────────
    if list_mode:
        if not hints_file.exists():
            print("  No .reflect_hints.json found — no hints captured yet.", file=sys.stderr)
            return
        try:
            hints = json.loads(hints_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("  Warning: .reflect_hints.json is malformed.", file=sys.stderr)
            return
        if not hints:
            print("  .reflect_hints.json is empty.", file=sys.stderr)
            return
        print(f"  {len(hints)} hint(s) in {hints_file}:\n", file=sys.stderr)
        for i, h in enumerate(hints, 1):
            phase_str = f"Phase {h['phase']}" if "phase" in h else "Phase ?"
            type_str = h.get("type_hint", "unclassified")
            ts = h.get("timestamp", "")
            print(f"  [{i}] {phase_str} ({type_str}) {ts}", file=sys.stderr)
            print(f"       {h['hint']}", file=sys.stderr)
        return

    # ── Append mode ────────────────────────────────────────────────────────────
    if not hint:
        print("Error: --hint is required (or use --list to see existing hints).", file=sys.stderr)
        sys.exit(1)

    # Read → append → overwrite (never clobber prior hints)
    hints = []
    if hints_file.exists():
        try:
            existing = json.loads(hints_file.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                hints = existing
        except json.JSONDecodeError:
            pass  # start fresh if file is malformed

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "hint": hint,
    }
    if phase is not None:
        entry["phase"] = phase
    if type_hint:
        entry["type_hint"] = type_hint

    hints.append(entry)
    hints_file.write_text(
        json.dumps(hints, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"  Hint appended to {hints_file} ({len(hints)} total hint(s))",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
