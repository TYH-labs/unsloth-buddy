"""
reflect.py — Long-term memory extraction and persistence for unsloth-buddy.

This is a stdlib-only, dumb file-management tool.  It does NO LLM work.
The agent that invokes it IS the LLM — it classifies and summarises.

Two modes:

  # Mode 1: Extract raw candidates from a completed project
  python scripts/reflect.py path/to/project_dir --extract
  python scripts/reflect.py --extract --all          # scan cwd for *_YYYY_MM_DD/ dirs

  # Mode 2: Write agent-classified entries to ~/.gaslamp/
  echo '<json>' | python scripts/reflect.py --write
  echo '<json>' | python scripts/reflect.py --write --dry-run

Extract reads gaslamp.md (§5, §6, §9, §11) + memory.md (Discoveries) and
emits structured JSON to stdout.  Write reads classified JSON from stdin and
merges it into ~/.gaslamp/{user,lessons,skills}.md with dedup, char-limit
enforcement, and quarterly archiving.
"""

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Constants ────────────────────────────────────────────────────────────────

GASLAMP_HOME = Path.home() / ".gaslamp"
CHAR_LIMITS = {"user": 2000, "lessons": 3000, "skills": 3000}
SCHEMA_VERSION = "1.0"
SCHEMAS = {
    "user": f"gaslamp-user/{SCHEMA_VERSION}",
    "lessons": f"gaslamp-lessons/{SCHEMA_VERSION}",
    "skills": f"gaslamp-skills/{SCHEMA_VERSION}",
}

# Regex for project dirs: <name>_YYYY_MM_DD
PROJECT_DIR_RE = re.compile(r"^.+_\d{4}_\d{2}_\d{2}$")

# gaslamp.md section headers (## N. Title)
SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(.+)$", re.MULTILINE)

# 📖 Learn blocks to skip
LEARN_BLOCK_RE = re.compile(
    r"^\s*>\s*📖\s*\*\*Learn.*?\n(?:\s*>.*\n)*", re.MULTILINE
)

# Entry header in memory files: ### [YYYY-MM-DD] Title
ENTRY_HEADER_RE = re.compile(r"^###\s+\[(\d{4}-\d{2}-\d{2})\]\s+(.+)$", re.MULTILINE)


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACT MODE
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_gaslamp_sections(text: str) -> dict[str, str]:
    """Parse gaslamp.md into {section_number: raw_text} dict."""
    matches = list(SECTION_RE.finditer(text))
    sections = {}
    for i, m in enumerate(matches):
        num = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # Strip 📖 Learn blocks — generic education, not operational lessons
        body = LEARN_BLOCK_RE.sub("", body).strip()
        sections[num] = body
    return sections


def _parse_memory_discoveries(text: str) -> str:
    """Extract non-placeholder lines from memory.md Discoveries & Notes."""
    in_discoveries = False
    lines = []
    for line in text.splitlines():
        if "Discoveries" in line and "Notes" in line:
            in_discoveries = True
            continue
        if in_discoveries:
            # Stop at next heading
            if line.startswith("#"):
                break
            stripped = line.strip()
            # Skip blanks, HTML comments, placeholder lines
            if (
                not stripped
                or stripped.startswith("<!--")
                or stripped.endswith("-->")
                or "TBD" in stripped
            ):
                continue
            lines.append(stripped)
    return "\n".join(lines)


def _infer_project_date(dirname: str) -> str:
    """Extract YYYY-MM-DD from a project dir name like 'qwen_chip2_sft_2026_03_17'."""
    parts = dirname.rsplit("_", 3)
    if len(parts) >= 4:
        try:
            y, m, d = parts[-3], parts[-2], parts[-1]
            return f"{y}-{m}-{d}"
        except (ValueError, IndexError):
            pass
    return datetime.now().strftime("%Y-%m-%d")


def extract_project(project_dir: Path) -> Optional[Dict]:
    """Extract candidates from a single project directory."""
    gaslamp_path = project_dir / "gaslamp.md"
    memory_path = project_dir / "memory.md"

    if not gaslamp_path.exists():
        return None

    project_name = project_dir.name
    project_date = _infer_project_date(project_name)

    candidates = []

    # Parse gaslamp.md sections
    gaslamp_text = gaslamp_path.read_text(encoding="utf-8")
    sections = _parse_gaslamp_sections(gaslamp_text)

    # § 5 Environment
    if "5" in sections and sections["5"].strip():
        candidates.append({
            "section": "environment",
            "source": "gaslamp.md §5",
            "text": sections["5"],
        })

    # § 6 Hyperparameters
    if "6" in sections and sections["6"].strip():
        candidates.append({
            "section": "hyperparameters",
            "source": "gaslamp.md §6",
            "text": sections["6"],
        })

    # § 9 File Inventory
    if "9" in sections and sections["9"].strip():
        candidates.append({
            "section": "file_inventory",
            "source": "gaslamp.md §9",
            "text": sections["9"],
        })

    # § 11 Workarounds & Critical Notes
    if "11" in sections and sections["11"].strip():
        candidates.append({
            "section": "workarounds",
            "source": "gaslamp.md §11",
            "text": sections["11"],
        })

    # memory.md Discoveries & Notes
    if memory_path.exists():
        memory_text = memory_path.read_text(encoding="utf-8")
        discoveries = _parse_memory_discoveries(memory_text)
        if discoveries:
            candidates.append({
                "section": "discoveries",
                "source": "memory.md",
                "text": discoveries,
            })

    if not candidates:
        return None

    return {
        "project": project_name,
        "project_date": project_date,
        "candidates": candidates,
    }


def extract_all(base_dir: Path) -> List[Dict]:
    """Scan base_dir for *_YYYY_MM_DD/ project dirs and extract from each."""
    results = []
    for child in sorted(base_dir.iterdir()):
        if child.is_dir() and PROJECT_DIR_RE.match(child.name):
            result = extract_project(child)
            if result:
                results.append(result)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE MODE
# ═══════════════════════════════════════════════════════════════════════════════

def _content_hash(text: str) -> str:
    """sha256 of normalised text for dedup."""
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def _make_front_matter(schema: str, char_count: int) -> str:
    """Generate YAML front-matter block."""
    today = datetime.now().strftime("%Y-%m-%d")
    return f"---\nschema: {schema}\nupdated: {today}\nchar_count: {char_count}\n---\n"


def _parse_entries(text: str) -> List[Dict]:
    """Parse a memory file into a list of {date, title, body, hash} entries."""
    # Strip front-matter
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:].strip()

    entries = []
    matches = list(ENTRY_HEADER_RE.finditer(text))
    for i, m in enumerate(matches):
        date = m.group(1)
        title = m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        entries.append({
            "date": date,
            "title": title,
            "body": body,
            "hash": _content_hash(f"{title}\n{body}"),
        })
    return entries


def _quarter_for_date(date_str: str) -> str:
    """Return YYYY_Q{N} for a date string."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        dt = datetime.now()
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}_Q{q}"


def _render_entries(entries: list[dict]) -> str:
    """Render entries back to markdown body (no front-matter)."""
    parts = []
    for e in entries:
        parts.append(f"### [{e['date']}] {e['title']}")
        parts.append(e["body"])
        parts.append("")
    return "\n".join(parts).strip()


def _render_file(file_type: str, entries: list[dict]) -> str:
    """Render a full memory file with front-matter."""
    body = _render_entries(entries)
    char_count = len(body)
    front_matter = _make_front_matter(SCHEMAS[file_type], char_count)
    return f"{front_matter}\n{body}\n"


def _format_lesson_entry(item: dict) -> dict:
    """Format a classified lesson item into an entry."""
    body_parts = [item["body"]]
    if item.get("source"):
        body_parts.append(f"Source: {item['source']}")
    body = "\n".join(body_parts)
    return {
        "date": item.get("date", datetime.now().strftime("%Y-%m-%d")),
        "title": item["title"],
        "body": body,
        "hash": _content_hash(f"{item['title']}\n{body}"),
    }


def _format_user_entry(item: dict) -> dict:
    """Format a classified user preference item into an entry."""
    return {
        "date": item.get("date", datetime.now().strftime("%Y-%m-%d")),
        "title": item["title"],
        "body": item["body"],
        "hash": _content_hash(f"{item['title']}\n{item['body']}"),
    }


def _format_skill_entry(item: dict) -> dict:
    """Format a classified skill (recipe) item into an entry."""
    body_parts = []
    if item.get("when"):
        body_parts.append(f"When: {item['when']}")
    for step in item.get("steps", []):
        body_parts.append(f"- {step}")
    if item.get("source"):
        body_parts.append(f"Source: {item['source']}")
    body = "\n".join(body_parts)
    return {
        "date": item.get("date", datetime.now().strftime("%Y-%m-%d")),
        "title": item["title"],
        "body": body,
        "hash": _content_hash(f"{item['title']}\n{body}"),
    }


FORMATTERS = {
    "user": _format_user_entry,
    "lessons": _format_lesson_entry,
    "skills": _format_skill_entry,
}


def _ensure_gaslamp_home():
    """Create ~/.gaslamp/ with README.md if it doesn't exist."""
    GASLAMP_HOME.mkdir(exist_ok=True)
    (GASLAMP_HOME / "archive").mkdir(exist_ok=True)

    readme_path = GASLAMP_HOME / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# ~/.gaslamp/ — Long-Term Memory for Gaslamp Agents\n\n"
            "This directory is auto-managed by `scripts/reflect.py`.\n"
            "It stores cross-project knowledge that accumulates over time.\n\n"
            "| File | Role |\n"
            "|------|------|\n"
            "| `user.md` | Hardware profile, preferences, deploy targets (≤2000 chars) |\n"
            "| `lessons.md` | Model gotchas, install traps, workarounds (≤3000 chars) |\n"
            "| `skills.md` | Scenario recipes with trigger conditions (≤3000 chars) |\n"
            "| `archive/` | Evicted entries when char limits are exceeded |\n"
            "| `index.json` | Machine-readable manifest |\n\n"
            f"Schema version: {SCHEMA_VERSION}\n",
            encoding="utf-8",
        )


def _evict_oldest(
    entries: List[Dict], char_limit: int, file_type: str, dry_run: bool
) -> List[Dict]:
    """Remove oldest entries until body fits under char_limit. Archive evicted."""
    evicted = []
    while entries and len(_render_entries(entries)) > char_limit:
        evicted.append(entries.pop(0))  # oldest first (entries are date-sorted)

    if evicted and not dry_run:
        # Archive evicted entries by quarter
        by_quarter: dict[str, list[dict]] = {}
        for e in evicted:
            q = _quarter_for_date(e["date"])
            by_quarter.setdefault(q, []).append(e)

        archive_dir = GASLAMP_HOME / "archive"
        for quarter, quarter_entries in by_quarter.items():
            archive_path = archive_dir / f"{file_type}_{quarter}.md"
            existing = ""
            if archive_path.exists():
                existing = archive_path.read_text(encoding="utf-8")
            new_content = _render_entries(quarter_entries)
            if existing:
                archive_path.write_text(
                    f"{existing}\n\n{new_content}\n", encoding="utf-8"
                )
            else:
                archive_path.write_text(f"{new_content}\n", encoding="utf-8")

    if evicted:
        verb = "Would evict" if dry_run else "Evicted"
        for e in evicted:
            print(
                f"  {verb}: [{e['date']}] {e['title']} → archive/",
                file=sys.stderr,
            )

    return entries


def _update_index(file_stats: dict[str, dict]):
    """Write/update ~/.gaslamp/index.json."""
    index_path = GASLAMP_HOME / "index.json"
    index = {
        "schema_version": SCHEMA_VERSION,
        "updated": datetime.now().strftime("%Y-%m-%d"),
        "files": file_stats,
    }
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_entries(payload: dict, dry_run: bool = False):
    """Merge classified entries into ~/.gaslamp/ files."""
    if not dry_run:
        _ensure_gaslamp_home()

    file_stats = {}

    for file_type in ("user", "lessons", "skills"):
        new_items = payload.get(file_type, [])
        if not new_items:
            continue

        formatter = FORMATTERS[file_type]
        file_path = GASLAMP_HOME / f"{file_type}.md"

        # Load existing entries
        existing_entries = []
        existing_hashes = set()
        if file_path.exists():
            existing_entries = _parse_entries(
                file_path.read_text(encoding="utf-8")
            )
            existing_hashes = {e["hash"] for e in existing_entries}

        # Format and dedup new entries
        added = 0
        skipped = 0
        for item in new_items:
            entry = formatter(item)
            if entry["hash"] in existing_hashes:
                skipped += 1
                continue
            existing_entries.append(entry)
            existing_hashes.add(entry["hash"])
            added += 1

        if added == 0:
            if skipped:
                print(
                    f"  {file_type}.md: {skipped} duplicate(s) skipped, no new entries",
                    file=sys.stderr,
                )
            continue

        # Sort by date (oldest first, for eviction ordering)
        existing_entries.sort(key=lambda e: e["date"])

        # Enforce char limit via eviction
        char_limit = CHAR_LIMITS[file_type]
        existing_entries = _evict_oldest(
            existing_entries, char_limit, file_type, dry_run
        )

        # Render and write
        content = _render_file(file_type, existing_entries)
        body_len = len(_render_entries(existing_entries))

        if dry_run:
            print(f"\n  [DRY RUN] {file_type}.md:", file=sys.stderr)
            print(f"    +{added} new, {skipped} dup skipped", file=sys.stderr)
            print(f"    char_count: {body_len}/{char_limit}", file=sys.stderr)
            print(f"    entries: {len(existing_entries)}", file=sys.stderr)
        else:
            file_path.write_text(content, encoding="utf-8")
            print(
                f"  {file_type}.md: +{added} new, {skipped} dup skipped "
                f"({body_len}/{char_limit} chars, {len(existing_entries)} entries)",
                file=sys.stderr,
            )

        file_stats[file_type] = {
            "char_count": body_len,
            "updated": datetime.now().strftime("%Y-%m-%d"),
            "entry_count": len(existing_entries),
        }

    if file_stats and not dry_run:
        _update_index(file_stats)
        print(f"\n  Updated {GASLAMP_HOME}/index.json", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        print(__doc__, file=sys.stderr)
        sys.exit(0)

    # ── Extract mode ──────────────────────────────────────────────────────────
    if "--extract" in args:
        scan_all = "--all" in args
        if scan_all:
            results = extract_all(Path.cwd())
            if not results:
                print("No project directories found in cwd.", file=sys.stderr)
                sys.exit(1)
        else:
            # Find the project dir argument (not a flag)
            project_arg = None
            for a in args:
                if not a.startswith("-"):
                    project_arg = a
                    break
            if not project_arg:
                print(
                    "Usage: python scripts/reflect.py <project_dir> --extract",
                    file=sys.stderr,
                )
                sys.exit(1)
            project_dir = Path(project_arg)
            if not project_dir.is_dir():
                print(f"Not a directory: {project_dir}", file=sys.stderr)
                sys.exit(1)
            result = extract_project(project_dir)
            if not result:
                print(f"No gaslamp.md found in {project_dir}", file=sys.stderr)
                sys.exit(1)
            results = [result]

        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # ── Write mode ────────────────────────────────────────────────────────────
    if "--write" in args:
        dry_run = "--dry-run" in args

        # Read JSON from stdin
        if sys.stdin.isatty():
            print(
                "Error: --write expects JSON on stdin.\n"
                "Usage: echo '<json>' | python scripts/reflect.py --write",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            payload = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON on stdin: {e}", file=sys.stderr)
            sys.exit(1)

        label = "[DRY RUN] " if dry_run else ""
        print(f"\n{label}Reflecting to {GASLAMP_HOME}/\n", file=sys.stderr)
        write_entries(payload, dry_run=dry_run)
        if not dry_run:
            print(f"\n  Done. Memory updated at {GASLAMP_HOME}/", file=sys.stderr)
        return

    # ── Unknown ───────────────────────────────────────────────────────────────
    print(
        "Unknown arguments. Use --extract or --write. See --help.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
