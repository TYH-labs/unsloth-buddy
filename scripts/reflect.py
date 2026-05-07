"""
reflect.py — Long-term memory extraction and persistence for unsloth-buddy.

This is a stdlib-only, dumb file-management tool.  It does NO LLM work.
The agent that invokes it IS the LLM — it classifies and summarises.

Three modes:

  # Mode 1: Extract raw candidates from a completed project
  python scripts/reflect.py path/to/project_dir --extract
  python scripts/reflect.py --extract --all          # scan cwd for *_YYYY_MM_DD/ dirs

  # Mode 2: Write agent-classified entries to ~/.gaslamp/
  echo '<json>' | python scripts/reflect.py --write
  echo '<json>' | python scripts/reflect.py --write --dry-run
  python scripts/reflect.py --write --input payload.json
  python scripts/reflect.py --write --input payload.json --dry-run
  python scripts/reflect.py --write --input payload.json --decisions-out .reflect_decisions.md
  python scripts/reflect.py --write --input payload.json --gaslamp-home /custom/path

  # Mode 3: Sandboxed backtest (copies ~/.gaslamp/ to .tmp/ sandbox, no live writes)
  python scripts/reflect.py --write --input payload.json --backtest <label>

Extract reads gaslamp.md (§5, §6, §9, §11) + memory.md (Discoveries) and
emits structured JSON to stdout.  Write reads classified JSON from stdin and
merges it into ~/.gaslamp/{user,lessons,skills}.md with dedup, char-limit
enforcement, and quarterly archiving.

--gaslamp-home <path>  overrides the default ~/.gaslamp/ target for all writes.
  Safety note: this is a redirect, not a filesystem lock.  The live tree is
  protected only by the caller supplying the correct path.

--backtest <label>  copies ~/.gaslamp/ to .tmp/reflect-backtests/<label>/gaslamp/,
  runs the write against that copy, prints a unified diff, and confirms the
  live tree was not touched.

--decisions-out <file>  writes a .reflect_decisions.md audit trail of every
  incoming entry (promoted / category_replaced / dup_skipped / validation_skipped)
  and every eviction (evicted_to_archive).  Defaults to <cwd>/.reflect_decisions.md
  when <cwd>/gaslamp.md exists; omitted otherwise.
"""

import difflib
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Category header in user.md (new format): ## Category Name
USER_CATEGORY_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


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


# Standard memory.md template sections — skip these, extract everything else
_MEMORY_SKIP_HEADERS = frozenset({"Model", "Dataset", "Hyperparameters"})


def _parse_memory_discoveries(text: str) -> str:
    """Extract content from all non-standard ## sections in memory.md.

    Standard template sections (Model, Dataset, Hyperparameters) are skipped.
    Everything else — 'Discoveries & Notes', 'Template friction log',
    'Dashboard issues', or any custom header — is treated as discovery content.
    This tolerates agents who use descriptive section names instead of the
    canonical 'Discoveries & Notes' header.
    """
    lines = []
    in_custom_section = False

    for line in text.splitlines():
        # ## section boundary — determine if standard or custom
        if line.startswith("## "):
            # Strip trailing parenthetical or dash-delimited commentary
            # e.g. "Template friction log — mps_grpo_example.py" → "Template friction log"
            raw_name = line[3:].strip()
            base_name = raw_name.split("(")[0].split("—")[0].strip()
            in_custom_section = base_name not in _MEMORY_SKIP_HEADERS
            continue

        # H1 title line — skip
        if line.startswith("# "):
            continue

        if not in_custom_section:
            continue

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


def _is_placeholder_section(text: str) -> bool:
    """Return True if a gaslamp.md section contains only template noise.

    Detects: HTML comment blocks, empty table rows (| | |), table rows where
    all value cells are blank (| Parameter | | |), and section separators (---).
    Sections that pass this check are skipped from extract candidates.
    """
    for line in text.splitlines():
        s = line.strip()
        if not s or s == "---":
            continue
        if s.startswith("<!--") or s.endswith("-->"):
            continue
        # Table separator row: |---|---|
        if s.startswith("|") and all(c in "|-: " for c in s):
            continue
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s[1:-1].split("|")]
            # All cells empty: | | |
            if all(not c for c in cells):
                continue
            # Header cell present but all value cells empty: | Parameter | | |
            if len(cells) >= 2 and all(not c for c in cells[1:]):
                continue
            # Template column header row: all cells are short plain words with no
            # backticks, slashes, or digits (e.g. | Parameter | Value | Why |)
            if all(
                len(c) <= 20 and "`" not in c and "/" not in c
                and not any(ch.isdigit() for ch in c)
                for c in cells if c
            ):
                continue
        # Any other non-empty line is meaningful content
        return False
    return True  # only noise found


def _candidate_id(project_name: str, section: str, text: str) -> str:
    """Stable id for a candidate: <project>:<section>:<sha8 of text>.

    Deterministic — same project + section + unchanged text always produces
    the same id.  Used to link payload entries back to the raw candidate.
    """
    sha8 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{project_name}:{section}:{sha8}"


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
    if "5" in sections and sections["5"].strip() and not _is_placeholder_section(sections["5"]):
        candidates.append({
            "section": "environment",
            "source": "gaslamp.md §5",
            "text": sections["5"],
        })

    # § 6 Hyperparameters
    if "6" in sections and sections["6"].strip() and not _is_placeholder_section(sections["6"]):
        candidates.append({
            "section": "hyperparameters",
            "source": "gaslamp.md §6",
            "text": sections["6"],
        })

    # § 9 File Inventory
    if "9" in sections and sections["9"].strip() and not _is_placeholder_section(sections["9"]):
        candidates.append({
            "section": "file_inventory",
            "source": "gaslamp.md §9",
            "text": sections["9"],
        })

    # § 11 Workarounds & Critical Notes
    if "11" in sections and sections["11"].strip() and not _is_placeholder_section(sections["11"]):
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

    # .reflect_hints.json — inline hints written during the session
    # These are pre-flagged by the agent at the moment of discovery, so they
    # don't need archaeological reconstruction in Phase 7.
    hints_path = project_dir / ".reflect_hints.json"
    if hints_path.exists():
        try:
            hints = json.loads(hints_path.read_text(encoding="utf-8"))
            if isinstance(hints, list) and hints:
                # Render hints as human-readable text for the extract output
                hint_lines = []
                for h in hints:
                    phase = h.get("phase", "?")
                    type_hint = h.get("type_hint", "?")
                    hint_text = h.get("hint", "")
                    hint_lines.append(f"[Phase {phase}] ({type_hint}) {hint_text}")
                candidates.append({
                    "section": "inline_hints",
                    "source": ".reflect_hints.json",
                    "pre_flagged": True,
                    "hints": hints,
                    "text": "\n".join(hint_lines),
                })
        except (json.JSONDecodeError, AttributeError):
            pass  # malformed hints file — skip silently

    if not candidates:
        return None

    # Stamp each candidate with a stable id (P1-2)
    for c in candidates:
        c["candidate_id"] = _candidate_id(project_name, c["section"], c["text"])

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


def _hash_body(title: str, body: str) -> str:
    """Dedup hash for an entry, excluding Source-ids metadata lines.

    Source-ids lines are traceability metadata added by P1-2; they must not
    affect the dedup hash so that entries written with or without source_candidates
    are correctly detected as duplicates.
    """
    clean_lines = [l for l in body.splitlines() if not l.startswith("Source-ids:")]
    return _content_hash(f"{title}\n" + "\n".join(clean_lines))


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
            "hash": _hash_body(title, body),
        })
    return entries


def _parse_user_entries(text: str) -> List[Dict]:
    """Parse user.md into a list of {category, body, date} entries.

    Handles the new ## Category format. If the file still uses the old
    ### [YYYY-MM-DD] Title format (schema v1.0), auto-migrates by splitting
    on ' — ' to derive a category name — the caller will rewrite in new format.
    """
    # Strip front-matter
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:].strip()

    # Try new ## Category format
    matches = list(USER_CATEGORY_RE.finditer(text))
    if matches:
        entries = []
        for i, m in enumerate(matches):
            category = m.group(1).strip()
            start = m.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[start:end_pos].strip()
            date = datetime.now().strftime("%Y-%m-%d")
            body_lines = []
            for line in block.splitlines():
                if line.startswith("Updated: "):
                    date = line.replace("Updated: ", "").strip()
                else:
                    body_lines.append(line)
            body = "\n".join(body_lines).strip()
            entries.append({"category": category, "body": body, "date": date})
        return entries

    # Fallback: auto-migrate from old ### [YYYY-MM-DD] Title format
    migrated = []
    for e in _parse_entries(text):
        # "Hardware — Apple Silicon" → "Hardware"
        category = e["title"].split("—")[0].strip()
        migrated.append({"category": category, "body": e["body"], "date": e["date"]})
    return migrated


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


def _render_user_entries(entries: List[Dict]) -> str:
    """Render user.md entries as ## Category sections with Updated: date lines."""
    parts = []
    for e in entries:
        parts.append(f"## {e['category']}")
        parts.append(e["body"])
        parts.append(f"Updated: {e['date']}")
        parts.append("")
    return "\n".join(parts).strip()


def _render_user_file(entries: List[Dict]) -> str:
    """Render a full user.md with front-matter."""
    body = _render_user_entries(entries)
    char_count = len(body)
    front_matter = _make_front_matter(SCHEMAS["user"], char_count)
    return f"{front_matter}\n{body}\n"


def _format_lesson_entry(item: dict) -> dict:
    """Format a classified lesson item into an entry."""
    body_parts = [item["body"]]
    if item.get("source"):
        body_parts.append(f"Source: {item['source']}")
    # Hash computed before Source-ids so dedup is stable across writes with/without ids
    hash_body = "\n".join(body_parts)
    if item.get("source_candidates"):
        body_parts.append(f"Source-ids: {', '.join(item['source_candidates'])}")
    body = "\n".join(body_parts)
    return {
        "date": item.get("date", datetime.now().strftime("%Y-%m-%d")),
        "title": item["title"],
        "body": body,
        "hash": _content_hash(f"{item['title']}\n{hash_body}"),
    }


def _format_user_entry(item: dict) -> dict:
    """Format a classified user preference item into a category entry."""
    return {
        "category": item["title"],  # title field becomes the ## Category heading
        "body": item["body"],
        "date": item.get("date", datetime.now().strftime("%Y-%m-%d")),
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
    # Hash computed before Source-ids so dedup is stable across writes with/without ids
    hash_body = "\n".join(body_parts)
    if item.get("source_candidates"):
        body_parts.append(f"Source-ids: {', '.join(item['source_candidates'])}")
    body = "\n".join(body_parts)
    return {
        "date": item.get("date", datetime.now().strftime("%Y-%m-%d")),
        "title": item["title"],
        "body": body,
        "hash": _content_hash(f"{item['title']}\n{hash_body}"),
    }


FORMATTERS = {
    "user": _format_user_entry,
    "lessons": _format_lesson_entry,
    "skills": _format_skill_entry,
}


_VALID_PRIORITY = {"low", "medium", "high"}
_VALID_DURABILITY = {"transient", "recurring", "durable"}
_VALID_DECISION = {"promote", "keep_project_only"}


def _passes_promotion_gate(item: dict) -> tuple[bool, str]:
    """Return (True, '') if item passes the promotion gate, (False, reason) otherwise.

    Gate fields (priority, durability, decision) are optional — absent fields
    pass through for backwards compatibility with v1.1 payloads.  When present,
    all three must satisfy: priority=high AND durability=durable AND decision=promote.
    """
    priority = item.get("priority")
    durability = item.get("durability")
    decision = item.get("decision")

    if priority is not None and priority not in _VALID_PRIORITY:
        return False, f"unknown priority={priority!r}"
    if durability is not None and durability not in _VALID_DURABILITY:
        return False, f"unknown durability={durability!r}"
    if decision is not None and decision not in _VALID_DECISION:
        return False, f"unknown decision={decision!r}"

    if priority is not None and priority != "high":
        return False, f"priority={priority} (need high)"
    if durability is not None and durability != "durable":
        return False, f"durability={durability} (need durable)"
    if decision == "keep_project_only":
        return False, "decision=keep_project_only"
    return True, ""


def _validate_item(file_type: str, item: dict) -> List[str]:
    """Return validation error strings for a payload item. Empty list = valid.

    F-7: prevents missing required fields from writing silently malformed entries.
    """
    errors = []
    if file_type in ("lessons", "user"):
        if not str(item.get("title", "")).strip():
            errors.append("missing 'title'")
        if not str(item.get("body", "")).strip():
            errors.append("missing 'body'")
    elif file_type == "skills":
        if not str(item.get("title", "")).strip():
            errors.append("missing 'title'")
        if not str(item.get("when", "")).strip():
            errors.append("missing 'when'")
        steps = item.get("steps")
        if not steps or not isinstance(steps, list) or len(steps) == 0:
            errors.append("missing or empty 'steps' (must be a non-empty list)")
    return errors


def _ensure_gaslamp_home(gaslamp_home: Path):
    """Create gaslamp_home with README.md if it doesn't exist."""
    gaslamp_home.mkdir(exist_ok=True)
    (gaslamp_home / "archive").mkdir(exist_ok=True)

    readme_path = gaslamp_home / "README.md"
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
    entries: List[Dict],
    char_limit: int,
    file_type: str,
    dry_run: bool,
    gaslamp_home: Path,
) -> Tuple[List[Dict], List[Dict]]:
    """Remove oldest entries until body fits under char_limit. Archive evicted.

    Returns (remaining_entries, evicted_entries).
    """
    evicted = []
    while entries and len(_render_entries(entries)) > char_limit:
        evicted.append(entries.pop(0))  # oldest first (entries are date-sorted)

    if evicted and not dry_run:
        # Archive evicted entries by quarter
        by_quarter: dict[str, list[dict]] = {}
        for e in evicted:
            q = _quarter_for_date(e["date"])
            by_quarter.setdefault(q, []).append(e)

        archive_dir = gaslamp_home / "archive"
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

    return entries, evicted


def _update_index(file_stats: dict[str, dict], gaslamp_home: Path):
    """Write/update gaslamp_home/index.json."""
    index_path = gaslamp_home / "index.json"
    index = {
        "schema_version": SCHEMA_VERSION,
        "updated": datetime.now().strftime("%Y-%m-%d"),
        "files": file_stats,
    }
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_decisions_md(decisions: List[Dict], out_path: Path, payload: dict, gaslamp_home: Path):
    """Write a .reflect_decisions.md audit trail for the current write run."""
    today = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "# .reflect_decisions.md — Phase 7 Reflection Audit",
        "",
        f"Reflected: {today}  ",
        f"Target: `{gaslamp_home}`",
        "",
        "## Incoming Entries",
        "",
        "| Title | Type | Action | Reason |",
        "|-------|------|--------|--------|",
    ]

    incoming = [d for d in decisions if d["action"] != "evicted_to_archive"]
    evicted = [d for d in decisions if d["action"] == "evicted_to_archive"]

    for d in incoming:
        title = d["title"].replace("|", "\\|")
        reason = d.get("reason", "—").replace("|", "\\|")
        lines.append(f"| {title} | {d['file_type']} | {d['action']} | {reason} |")

    if not incoming:
        lines.append("| — | — | — | no entries processed |")

    if evicted:
        lines += [
            "",
            "## Evicted from Memory (char-limit pressure)",
            "",
            "| Title | Type | Action | Reason |",
            "|-------|------|--------|--------|",
        ]
        for d in evicted:
            title = d["title"].replace("|", "\\|")
            reason = d.get("reason", "evicted to archive/").replace("|", "\\|")
            lines.append(f"| {title} | {d['file_type']} | evicted_to_archive | {reason} |")

    # Summary counts
    promoted = sum(1 for d in incoming if d["action"] in ("promoted", "category_replaced"))
    skipped = sum(1 for d in incoming if d["action"] in ("dup_skipped", "validation_skipped", "gate_blocked"))
    lines += [
        "",
        "## Summary",
        "",
        f"- Promoted: {promoted}",
        f"- Skipped: {skipped}",
        f"- Evicted: {len(evicted)}",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_entries(
    payload: dict,
    gaslamp_home: Path = GASLAMP_HOME,
    dry_run: bool = False,
    decisions_out: Optional[Path] = None,
):
    """Merge classified entries into gaslamp_home/ files.

    gaslamp_home defaults to ~/.gaslamp/ but can be overridden for sandboxed
    backtest runs.  Safety relies on the caller passing the correct path — there
    is no filesystem-level lock on the live tree.
    """
    if not dry_run:
        _ensure_gaslamp_home(gaslamp_home)

    file_stats = {}
    decisions: List[Dict] = []

    # Backwards-compat warning: v1.1 payloads omit gate fields; warn once per run
    all_items = [
        item
        for ft in ("user", "lessons", "skills")
        for item in payload.get(ft, [])
    ]
    legacy_count = sum(
        1 for item in all_items
        if item.get("priority") is None
        and item.get("durability") is None
        and item.get("decision") is None
    )
    if legacy_count:
        print(
            f"  Note: v1.1 payload detected — gate fields absent for "
            f"{legacy_count} entr{'y' if legacy_count == 1 else 'ies'} "
            f"(treating as priority=high/durable/promote)",
            file=sys.stderr,
        )

    for file_type in ("user", "lessons", "skills"):
        new_items = payload.get(file_type, [])
        if not new_items:
            continue

        # F-7: validate all items before writing — skip malformed entries
        valid_items = []
        for item in new_items:
            errors = _validate_item(file_type, item)
            if errors:
                title = item.get("title", "(no title)")
                print(
                    f"  Warning: skipping {file_type} entry {title!r}: "
                    f"{', '.join(errors)}",
                    file=sys.stderr,
                )
                decisions.append({
                    "title": title,
                    "file_type": file_type,
                    "action": "validation_skipped",
                    "reason": "; ".join(errors),
                })
            else:
                valid_items.append(item)
        if not valid_items:
            continue

        # P1-1: promotion gate — filter on priority/durability/decision when present
        gated_items = []
        for item in valid_items:
            passes, reason = _passes_promotion_gate(item)
            if not passes:
                title = item.get("title", "(no title)")
                print(
                    f"  Gate: dropping {file_type} entry {title!r}: {reason}",
                    file=sys.stderr,
                )
                decisions.append({
                    "title": title,
                    "file_type": file_type,
                    "action": "gate_blocked",
                    "reason": reason,
                })
            else:
                gated_items.append(item)
        if not gated_items:
            continue
        new_items = gated_items

        file_path = gaslamp_home / f"{file_type}.md"

        # F-6: user.md uses category-replace logic (not append + SHA dedup)
        if file_type == "user":
            existing_entries = []
            if file_path.exists():
                # _parse_user_entries auto-migrates from old ### dated format
                existing_entries = _parse_user_entries(
                    file_path.read_text(encoding="utf-8")
                )

            existing_by_cat = {e["category"]: i for i, e in enumerate(existing_entries)}
            added = 0
            updated = 0
            for item in new_items:
                entry = _format_user_entry(item)
                cat = entry["category"]
                if cat in existing_by_cat:
                    existing_entries[existing_by_cat[cat]] = entry
                    updated += 1
                    decisions.append({
                        "title": cat,
                        "file_type": "user",
                        "action": "category_replaced",
                        "reason": "category updated in-place",
                    })
                else:
                    existing_by_cat[cat] = len(existing_entries)
                    existing_entries.append(entry)
                    added += 1
                    decisions.append({
                        "title": cat,
                        "file_type": "user",
                        "action": "promoted",
                        "reason": "—",
                    })

            if added == 0 and updated == 0:
                continue

            body_len = len(_render_user_entries(existing_entries))

            if dry_run:
                print(f"\n  [DRY RUN] user.md:", file=sys.stderr)
                print(f"    +{added} new, {updated} updated in-place", file=sys.stderr)
                print(f"    char_count: {body_len}/{CHAR_LIMITS['user']}", file=sys.stderr)
                print(f"    entries: {len(existing_entries)}", file=sys.stderr)
            else:
                file_path.write_text(
                    _render_user_file(existing_entries), encoding="utf-8"
                )
                print(
                    f"  user.md: +{added} new, {updated} updated in-place "
                    f"({body_len}/{CHAR_LIMITS['user']} chars, "
                    f"{len(existing_entries)} entries)",
                    file=sys.stderr,
                )

            file_stats["user"] = {
                "char_count": body_len,
                "updated": datetime.now().strftime("%Y-%m-%d"),
                "entry_count": len(existing_entries),
            }
            continue  # skip generic flow below

        # ── Generic flow for lessons and skills ───────────────────────────────
        formatter = FORMATTERS[file_type]

        existing_entries = []
        existing_hashes = set()
        if file_path.exists():
            existing_entries = _parse_entries(file_path.read_text(encoding="utf-8"))
            existing_hashes = {e["hash"] for e in existing_entries}

        added = 0
        skipped = 0
        for item in new_items:
            entry = formatter(item)
            if entry["hash"] in existing_hashes:
                skipped += 1
                decisions.append({
                    "title": item.get("title", "(no title)"),
                    "file_type": file_type,
                    "action": "dup_skipped",
                    "reason": "sha256 match with existing entry",
                })
                continue
            existing_entries.append(entry)
            existing_hashes.add(entry["hash"])
            added += 1
            decisions.append({
                "title": item.get("title", "(no title)"),
                "file_type": file_type,
                "action": "promoted",
                "reason": "—",
            })

        if added == 0:
            if skipped:
                print(
                    f"  {file_type}.md: {skipped} duplicate(s) skipped, no new entries",
                    file=sys.stderr,
                )
            continue

        existing_entries.sort(key=lambda e: e["date"])

        char_limit = CHAR_LIMITS[file_type]
        existing_entries, evicted = _evict_oldest(
            existing_entries, char_limit, file_type, dry_run, gaslamp_home
        )
        for e in evicted:
            quarter = _quarter_for_date(e["date"])
            decisions.append({
                "title": e["title"],
                "file_type": file_type,
                "action": "evicted_to_archive",
                "reason": f"archive/{file_type}_{quarter}.md",
            })

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
        _update_index(file_stats, gaslamp_home)
        print(f"\n  Updated {gaslamp_home}/index.json", file=sys.stderr)

    # Write decisions file (P0-2); dry-run emits a .preview.md so gate outcomes are visible
    if decisions:
        _resolve_and_write_decisions(
            decisions, payload, gaslamp_home, decisions_out, preview=dry_run
        )


def _resolve_and_write_decisions(
    decisions: List[Dict],
    payload: dict,
    gaslamp_home: Path,
    decisions_out: Optional[Path],
    preview: bool = False,
):
    """Write .reflect_decisions[.preview].md to the resolved output path.

    Resolution order:
    1. Explicit --decisions-out path (always honoured; .preview suffix added when preview=True)
    2. <cwd>/.reflect_decisions[.preview].md when <cwd>/gaslamp.md exists
    3. Skip — avoids polluting non-project directories
    """
    suffix = ".preview.md" if preview else ".md"
    if decisions_out is not None:
        # For an explicit path: strip any extension and re-apply correct suffix
        stem = decisions_out.with_suffix("").name
        if stem.endswith(".preview"):
            stem = stem[: -len(".preview")]
        out_path = decisions_out.parent / (stem + suffix)
    elif (Path.cwd() / "gaslamp.md").exists():
        out_path = Path.cwd() / f".reflect_decisions{suffix}"
    else:
        return  # not in a project dir and no explicit path

    _write_decisions_md(decisions, out_path, payload, gaslamp_home)
    tag = "[DRY RUN] " if preview else ""
    print(f"  {tag}Decisions: {out_path}", file=sys.stderr)


# ── Backtest ─────────────────────────────────────────────────────────────────

def _run_backtest(
    label: str,
    payload: dict,
    decisions_out: Optional[Path],
    gaslamp_home: Path = GASLAMP_HOME,
):
    """Copy gaslamp_home/ to a sandbox, run write there, report unified diff.

    Safety: all writes target the sandbox copy only.  The live gaslamp_home
    tree is never opened for writing during this function.
    """
    backtest_root = Path.cwd() / ".tmp" / "reflect-backtests" / label
    backtest_root.mkdir(parents=True, exist_ok=True)
    sandbox = backtest_root / "gaslamp"

    # Fresh copy of live tree into sandbox
    if sandbox.exists():
        shutil.rmtree(sandbox)

    if gaslamp_home.exists():
        shutil.copytree(gaslamp_home, sandbox)
        print(f"  Copied {gaslamp_home} → {sandbox}", file=sys.stderr)
    else:
        sandbox.mkdir(parents=True)
        print(f"  No {gaslamp_home} found — starting from empty sandbox", file=sys.stderr)

    # Snapshot before write
    target_files = ("user.md", "lessons.md", "skills.md", "index.json")
    before: dict[str, str] = {}
    for fname in target_files:
        p = sandbox / fname
        before[fname] = p.read_text(encoding="utf-8") if p.exists() else ""

    # Resolved decisions output
    if decisions_out is not None:
        bt_decisions_out = decisions_out
    else:
        bt_decisions_out = backtest_root / ".reflect_decisions.md"

    print(
        f"\n  [BACKTEST:{label}] Writing to sandbox: {sandbox}\n",
        file=sys.stderr,
    )

    # Run write against sandbox — NOT GASLAMP_HOME
    write_entries(payload, gaslamp_home=sandbox, dry_run=False, decisions_out=bt_decisions_out)

    # Diff report
    print("\n── Backtest diff ─────────────────────────────────", file=sys.stderr)
    any_diff = False
    for fname in target_files:
        p = sandbox / fname
        after_text = p.read_text(encoding="utf-8") if p.exists() else ""
        if before[fname] == after_text:
            continue
        any_diff = True
        diff = list(difflib.unified_diff(
            before[fname].splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=f"before/{fname}",
            tofile=f"after/{fname}",
        ))
        sys.stderr.writelines(diff)
        print("", file=sys.stderr)

    if not any_diff:
        print(
            "  No changes (all entries already present or payload empty).",
            file=sys.stderr,
        )

    print(
        f"\n  Live {gaslamp_home} NOT modified — all writes stayed in {sandbox}",
        file=sys.stderr,
    )
    if bt_decisions_out.exists():
        print(f"  Decisions: {bt_decisions_out}", file=sys.stderr)
    print(f"  Backtest artifacts: {backtest_root}", file=sys.stderr)


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

        # --gaslamp-home <path>
        gaslamp_home = GASLAMP_HOME
        if "--gaslamp-home" in args:
            idx = args.index("--gaslamp-home")
            if idx + 1 >= len(args):
                print("Error: --gaslamp-home requires a path argument.", file=sys.stderr)
                sys.exit(1)
            gaslamp_home = Path(args[idx + 1])

        # --decisions-out <file>
        decisions_out: Optional[Path] = None
        if "--decisions-out" in args:
            idx = args.index("--decisions-out")
            if idx + 1 >= len(args):
                print("Error: --decisions-out requires a file path argument.", file=sys.stderr)
                sys.exit(1)
            decisions_out = Path(args[idx + 1])

        # --backtest <label>
        backtest_label: Optional[str] = None
        if "--backtest" in args:
            idx = args.index("--backtest")
            if idx + 1 >= len(args):
                print("Error: --backtest requires a label argument.", file=sys.stderr)
                sys.exit(1)
            backtest_label = args[idx + 1]

        # --input <file>
        input_file = None
        if "--input" in args:
            idx = args.index("--input")
            if idx + 1 >= len(args):
                print("Error: --input requires a file path argument.", file=sys.stderr)
                sys.exit(1)
            input_file = Path(args[idx + 1])
            if not input_file.exists():
                print(f"Error: input file not found: {input_file}", file=sys.stderr)
                sys.exit(1)

        # Read JSON from file or stdin
        if input_file:
            try:
                payload = json.loads(input_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in {input_file}: {e}", file=sys.stderr)
                sys.exit(1)
        elif sys.stdin.isatty():
            print(
                "Error: --write expects JSON on stdin or via --input <file>.\n"
                "Usage: echo '<json>' | python scripts/reflect.py --write\n"
                "       python scripts/reflect.py --write --input payload.json",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            try:
                payload = json.load(sys.stdin)
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON on stdin: {e}", file=sys.stderr)
                sys.exit(1)

        if backtest_label:
            print(f"\n[BACKTEST:{backtest_label}] Sandbox run\n", file=sys.stderr)
            _run_backtest(backtest_label, payload, decisions_out, gaslamp_home=gaslamp_home)
            return

        label = "[DRY RUN] " if dry_run else ""
        print(f"\n{label}Reflecting to {gaslamp_home}/\n", file=sys.stderr)
        write_entries(payload, gaslamp_home=gaslamp_home, dry_run=dry_run, decisions_out=decisions_out)
        if not dry_run:
            print(f"\n  Done. Memory updated at {gaslamp_home}/", file=sys.stderr)
        return

    # ── Unknown ───────────────────────────────────────────────────────────────
    print(
        "Unknown arguments. Use --extract or --write. See --help.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
