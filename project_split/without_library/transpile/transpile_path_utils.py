from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Sequence


_EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    "bkup",
    "bakf",
}


def ensure_directory_on_syspath(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    target = resolved if resolved.is_dir() else resolved.parent
    target_str = str(target)
    if target_str not in sys.path:
        sys.path.insert(0, target_str)
    return target


def find_project_root(start: str | Path) -> Path:
    resolved = Path(start).resolve()
    current = resolved if resolved.is_dir() else resolved.parent

    for candidate in (current, *current.parents):
        if (candidate / "Transpile Algorithms").is_dir() and (candidate / "Theory Algorithms").is_dir():
            return candidate

    for candidate in (current, *current.parents):
        if (candidate / "Transpile Algorithms").is_dir() or (candidate / "Theory Algorithms").is_dir():
            return candidate

    return current


def _is_excluded(path: Path) -> bool:
    for part in path.parts:
        if part in _EXCLUDED_DIR_NAMES or part.startswith("[RESULT]"):
            return True
    return False


def _score_candidate(
    candidate: Path,
    *,
    root: Path,
    base_dir: Path,
    preferred_dirs: Sequence[str],
) -> tuple[int, int, str]:
    rel_parts = candidate.relative_to(root).parts
    score = 0

    if candidate.parent == base_dir:
        score += 120

    if rel_parts:
        top_dir = rel_parts[0]
        if top_dir in preferred_dirs:
            score += 300
        if top_dir == "Theory Algorithms":
            score += 100
        if top_dir == "Transpile Algorithms":
            score += 60

    score -= len(rel_parts)
    return (-score, len(rel_parts), str(candidate))


def resolve_project_file(
    start: str | Path,
    filename: str,
    *,
    preferred_dirs: Iterable[str] = (),
) -> Path:
    preferred_dirs = tuple(preferred_dirs)
    resolved = Path(start).resolve()
    base_dir = resolved if resolved.is_dir() else resolved.parent
    root = find_project_root(base_dir)

    for dirname in preferred_dirs:
        candidate = root / dirname / filename
        if candidate.is_file():
            return candidate

    direct_candidate = base_dir / filename
    if direct_candidate.is_file():
        return direct_candidate

    matches = []
    for candidate in root.rglob(filename):
        if not candidate.is_file():
            continue
        rel_candidate = candidate.relative_to(root)
        if _is_excluded(rel_candidate):
            continue
        matches.append(candidate)

    if not matches:
        preferred_hint = ", ".join(preferred_dirs) if preferred_dirs else "(no preferred directories)"
        raise FileNotFoundError(
            f"Could not find '{filename}' beneath project root '{root}'. Preferred directories: {preferred_hint}."
        )

    matches.sort(
        key=lambda candidate: _score_candidate(
            candidate,
            root=root,
            base_dir=base_dir,
            preferred_dirs=preferred_dirs,
        )
    )
    return matches[0]


def import_project_module(
    alias: str,
    start: str | Path,
    filename: str,
    *,
    preferred_dirs: Iterable[str] = (),
):
    module_path = resolve_project_file(start, filename, preferred_dirs=preferred_dirs)
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod, module_path

