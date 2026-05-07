"""HuggingFace dataset search for the data pipeline's Stage 2 (enrich)."""
from __future__ import annotations

import os
from typing import Sequence

from huggingface_hub import HfApi


def search_relevant_datasets(
    query: str,
    tags: Sequence[str] = (),
    limit: int = 10,
    token: str | None = None,
) -> list[dict]:
    """Search the Hub for datasets matching `query`, sorted by downloads.

    Returns a list of dicts: {id, downloads, tags, description}.
    """
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    # huggingface_hub 1.x uses `filter` instead of `tags`; keep the public
    # signature stable so callers don't need to track the rename.
    results = api.list_datasets(
        search=query,
        filter=list(tags) if tags else None,
        sort="downloads",
        limit=limit,
    )
    out = []
    for d in results:
        out.append({
            "id": getattr(d, "id", None),
            "downloads": getattr(d, "downloads", None),
            "tags": list(getattr(d, "tags", []) or []),
            "description": getattr(d, "description", None),
        })
    return out


if __name__ == "__main__":
    import json, sys
    q = " ".join(sys.argv[1:]) or "house prices"
    print(json.dumps(search_relevant_datasets(q, limit=5), indent=2, default=str))
