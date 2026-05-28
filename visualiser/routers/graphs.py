from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from visualiser.services.graph_builder import build_graphs

router = APIRouter()

SUPPORTED = ("python", "javascript", "java", "cpp")

SAMPLE_ROOTS: dict[str, Path] = {
    lang: Path(__file__).parent.parent.parent / "visualiser" / "samples" / lang
    for lang in SUPPORTED
}


@router.get("/graphs")
async def get_graphs(
    language: str = Query(default="python", description="Source language"),
):
    if language not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {language!r}. Choose from {SUPPORTED}",
        )

    sample_root = SAMPLE_ROOTS[language]
    if not sample_root.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Sample repo not found for language: {language}",
        )

    try:
        result = build_graphs(str(sample_root), language)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return result