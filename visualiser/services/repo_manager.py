from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import git


# ── Config ───────────────────────────────────────────────────────────────────

REPOS_BASE = Path(os.getenv("REPO_WORKSPACE", "/tmp/devgraph_repos"))
MAX_CACHED_REPOS = 5

LANGUAGE_EXTENSIONS = {
    "python":     {".py"},
    "javascript": {".js", ".jsx", ".ts", ".tsx"},
    "java":       {".java"},
    "cpp":        {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"},
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RepoInfo:
    url:       str
    name:      str
    local_path: str
    language:  str
    file_counts: dict[str, int]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _repo_name_from_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url.split("/")[-1]


def _detect_language(repo_path: Path) -> tuple[str, dict[str, int]]:
    skip_dirs = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "build", "dist", "target", ".gradle", ".idea", ".vscode",
    }

    counts: dict[str, int] = {lang: 0 for lang in LANGUAGE_EXTENSIONS}

    for root, dirs, files in os.walk(repo_path):
        # prune skip dirs in-place
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for fname in files:
            ext = Path(fname).suffix.lower()
            for lang, exts in LANGUAGE_EXTENSIONS.items():
                if ext in exts:
                    counts[lang] += 1

    dominant = max(counts, key=lambda l: counts[l])
    if counts[dominant] == 0:
        dominant = "python"  # safe default

    return dominant, counts


def _evict_oldest_repos(keep: int = MAX_CACHED_REPOS) -> None:
    if not REPOS_BASE.exists():
        return
    repos = sorted(
        [p for p in REPOS_BASE.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    for old in repos[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


# ── Public API ────────────────────────────────────────────────────────────────

def load_repo(url: str) -> RepoInfo:
    url = url.strip()
    if not url.startswith("http"):
        raise ValueError(f"Invalid URL: {url!r}. Must start with http/https.")

    name = _repo_name_from_url(url)
    REPOS_BASE.mkdir(parents=True, exist_ok=True)
    local_path = REPOS_BASE / name

    try:
        if local_path.exists():
            # already cloned — pull latest
            repo = git.Repo(local_path)
            origin = repo.remotes.origin
            origin.pull()
        else:
            _evict_oldest_repos(MAX_CACHED_REPOS - 1)
            git.Repo.clone_from(url, local_path, depth=1)  # shallow clone — faster
    except git.exc.GitCommandError as e:
        raise RuntimeError(f"Git operation failed: {e}") from e

    language, file_counts = _detect_language(local_path)

    return RepoInfo(
        url=url,
        name=name,
        local_path=str(local_path),
        language=language,
        file_counts=file_counts,
    )


def get_repo_path(name: str) -> Optional[Path]:
    p = REPOS_BASE / name
    return p if p.exists() else None


def list_cached_repos() -> list[str]:
    if not REPOS_BASE.exists():
        return []
    return [p.name for p in REPOS_BASE.iterdir() if p.is_dir()]
