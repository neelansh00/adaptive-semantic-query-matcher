"""Phase 11: documentation and startup checks.

  1. every relative Markdown link in README.md and docs/*.md resolves to an existing file or folder
  2. every repository file path named in backticks in README.md exists
  3. every module in src/ and scripts/ imports cleanly (scripts are imported, not run)

Usage:  python scripts/check_docs.py
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.data import PROJECT_ROOT  # noqa: E402

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
BACKTICK_PATH = re.compile(r"`((?:[\w.-]+/)+[\w.-]+\.(?:py|json|yaml|md|csv|npy|joblib|pt))`")


def check_links() -> list[str]:
    bad = []
    for md in [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").glob("*.md"))]:
        for target in LINK.findall(md.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = (md.parent / target.split("#")[0]).resolve()
            if not path.exists():
                bad.append(f"{md.relative_to(PROJECT_ROOT)} -> {target}")
    return bad


def check_readme_paths() -> list[str]:
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    return [p for p in set(BACKTICK_PATH.findall(text)) if not (PROJECT_ROOT / p).exists()]


def check_imports() -> list[str]:
    bad = []
    modules = [p for p in (PROJECT_ROOT / "src").rglob("*.py")] + [p for p in (PROJECT_ROOT / "scripts").glob("*.py")]
    for p in sorted(modules):
        name = ".".join(p.relative_to(PROJECT_ROOT).with_suffix("").parts)
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{name}: {type(e).__name__}: {e}")
    return bad


def main() -> None:
    links, paths, imports = check_links(), check_readme_paths(), check_imports()
    n_md = 1 + len(list((PROJECT_ROOT / "docs").glob("*.md")))
    print(f"markdown files checked: {n_md}; broken links: {links or 'none'}")
    print(f"README file paths missing: {paths or 'none'}")
    print(f"modules imported: {len(list((PROJECT_ROOT / 'src').rglob('*.py'))) + len(list((PROJECT_ROOT / 'scripts').glob('*.py')))}; "
          f"import errors: {imports or 'none'}")
    sys.exit(1 if links or paths or imports else 0)


if __name__ == "__main__":
    main()
