"""Ingest top public Kaggle notebooks for the active competition.

Downloads the top-5 most-voted public kernels, extracts feature engineering
insights with Ollama, and stores them in Postgres memory + a local markdown
file for quick reference by the agent.

Usage:
    python scripts/ingest_notebooks.py
    python scripts/ingest_notebooks.py --competition titanic
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.notify import _load_env, notify  # noqa: E402


def _active_competition() -> str:
    try:
        reg = json.loads((REPO_ROOT / "competitions" / "registry.json").read_text())
        return reg.get("active", "")
    except Exception:
        return ""


def _list_kernels(slug: str, n: int = 5) -> list[dict]:
    try:
        r = subprocess.run(
            ["kaggle", "kernels", "list", "--competition", slug,
             "--sort-by", "voteCount", "--page-size", str(n), "--csv"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            print(f"kaggle kernels list failed: {r.stderr}", file=sys.stderr)
            return []
        reader = csv.DictReader(io.StringIO(r.stdout.strip()))
        return list(reader)
    except Exception as e:
        print(f"_list_kernels error: {e}", file=sys.stderr)
        return []


def _pull_kernel(ref: str, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["kaggle", "kernels", "pull", ref, "-p", str(out_dir)],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            print(f"pull failed for {ref}: {r.stderr[:200]}", file=sys.stderr)
            return None
        # Find the downloaded notebook
        for f in out_dir.glob("*.ipynb"):
            return f
        for f in out_dir.glob("*.py"):
            return f
        return None
    except Exception as e:
        print(f"_pull_kernel error {ref}: {e}", file=sys.stderr)
        return None


def _extract_text(path: Path) -> str:
    try:
        if path.suffix == ".ipynb":
            nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            parts = []
            for cell in nb.get("cells", []):
                src = "".join(cell.get("source", []))
                if src.strip():
                    parts.append(src)
            return "\n\n".join(parts)
        else:
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"_extract_text error {path}: {e}", file=sys.stderr)
        return ""


def _analyse_with_ollama(slug: str, ref: str, content: str) -> str:
    try:
        from core.ollama_client import generate  # noqa: E402
        prompt = (
            f"You are analysing a top Kaggle notebook for the '{slug}' competition.\n"
            f"Kernel: {ref}\n\n"
            f"Extract and summarise concisely:\n"
            f"1. Key features engineered (be specific — column names, transformations)\n"
            f"2. Model architecture and hyperparameters\n"
            f"3. Preprocessing / data cleaning tricks\n"
            f"4. Any competition-specific insights or leaks discovered\n"
            f"5. CV / LB scores achieved\n\n"
            f"Notebook content (truncated):\n{content[:6000]}"
        )
        return generate(prompt, think=False, max_tokens=1024)
    except Exception as e:
        return f"[Ollama unavailable: {e}]"


def _pg_save(slug: str, ref: str, body: str) -> bool:
    try:
        import os
        import psycopg2  # type: ignore
        conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO memories (type, title, body, tags, created_at, updated_at)
               VALUES (%s, %s, %s, %s, NOW(), NOW())
               ON CONFLICT DO NOTHING""",
            ("notebook_insight", ref, body, [slug, "notebook"]),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"pg_save error: {e}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest top Kaggle notebooks")
    parser.add_argument("--competition", "-c", default="")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    _load_env()
    slug = args.competition or _active_competition()
    if not slug:
        print("No active competition. Use --competition <slug>", file=sys.stderr)
        return 1

    print(f"Fetching top {args.top} notebooks for: {slug}")
    kernels = _list_kernels(slug, args.top)
    if not kernels:
        print("No kernels found — check kaggle credentials.", file=sys.stderr)
        return 1

    out_dir = REPO_ROOT / "data" / "notebooks" / slug
    insights: list[str] = []

    for k in kernels:
        ref = k.get("ref", "").strip()
        title = k.get("title", ref).strip()
        votes = k.get("totalVotes", "?")
        if not ref:
            continue

        print(f"  Pulling: {ref} ({votes} votes)")
        nb_path = _pull_kernel(ref, out_dir / ref.replace("/", "_"))
        if not nb_path:
            continue

        content = _extract_text(nb_path)
        if not content.strip():
            continue

        print(f"  Analysing with Ollama...")
        insight = _analyse_with_ollama(slug, ref, content)

        section = f"## {title}\n**Ref:** {ref} | **Votes:** {votes}\n\n{insight}"
        insights.append(section)
        _pg_save(slug, ref, insight)
        print(f"  ✓ Saved insight for {ref}")

    if not insights:
        print("No insights extracted.", file=sys.stderr)
        return 1

    # Save combined markdown
    claude_dir = REPO_ROOT / ".claude"
    claude_dir.mkdir(exist_ok=True)
    md_path = claude_dir / f"notebook_insights_{slug}.md"
    md_path.write_text(
        f"# Public notebook insights — {slug}\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        + "\n\n---\n\n".join(insights)
    )
    print(f"\nInsights saved to: {md_path}")

    notify(f"📚 Ingested {len(insights)} public notebooks for {slug}. Feature ideas saved to memory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
