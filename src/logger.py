from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import RecommendationResult


class RunLogger:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None

    def log(self, result: RecommendationResult) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result.to_dict(),
        }
        with self.path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record) + "\n")
