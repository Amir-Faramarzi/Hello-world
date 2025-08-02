"""Configuration handling for redness tracker."""
from __future__ import annotations

import json
import logging
import logging.handlers
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml  # type: ignore[import-untyped]


LOGGER = logging.getLogger(__name__)


@dataclass
class Settings:
    """Application settings persisted to YAML."""

    roi: tuple[int, int, int, int] = (0, 0, 100, 100)
    use_max_red: bool = False

    @classmethod
    def load(cls, path: Path) -> "Settings":
        if path.exists():
            data = yaml.safe_load(path.read_text());
            data["roi"] = tuple(data.get("roi", (0, 0, 100, 100)))

            return cls(**data)
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self)))


def configure_logging(log_dir: Path) -> None:
    """Configure JSON logging with rotating file handler."""
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log", maxBytes=100_000, backupCount=5
    )
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    def emit_json(record: logging.LogRecord) -> str:
        return json.dumps({"level": record.levelname, "msg": record.getMessage()})

    handler.setFormatter(logging.Formatter(fmt="%(message)s"))
    original_emit = handler.emit

    def emit(record: logging.LogRecord) -> None:
        record.msg = emit_json(record)
        original_emit(record)

    handler.emit = emit  # type: ignore[method-assign]
