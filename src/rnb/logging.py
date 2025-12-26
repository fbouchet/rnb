# src/rnb/logging.py
from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

ROLE = str

# Single place to change colors/styles for roles (Rich styles)
DEFAULT_ROLE_STYLES: dict[ROLE, str] = {
    "user": "bold cyan",
    "agent": "bold green",
    "model": "magenta",
    "system": "yellow",
    "debug": "dim",
    "error": "bold red",
}


def set_role_styles(overrides: Mapping[ROLE, str]) -> None:
    """Override role->style mapping globally (single place)."""
    DEFAULT_ROLE_STYLES.update(dict(overrides))


class RoleFormatter(logging.Formatter):
    """
    Formatter that can optionally emit Rich markup for role-based colors.

    - If use_markup=True: wraps message with Rich markup based on record.role
    - If use_markup=False: emits plain text (suitable for file logs)
    """

    def __init__(
        self,
        *,
        show_role: bool = True,
        show_logger: bool = False,
        use_markup: bool = True,
    ):
        super().__init__()
        self.show_role = show_role
        self.show_logger = show_logger
        self.use_markup = use_markup

    def format(self, record: logging.LogRecord) -> str:
        role = getattr(record, "role", None)
        msg = record.getMessage()

        prefix_parts: list[str] = []
        if self.show_role and role:
            prefix_parts.append(f"[{role}]")
        if self.show_logger:
            prefix_parts.append(record.name)

        prefix = ""
        if prefix_parts:
            prefix = " ".join(prefix_parts) + " "

        if not self.use_markup:
            return f"{prefix}{msg}"

        style = DEFAULT_ROLE_STYLES.get(str(role), "")
        if style:
            # RichHandler has markup=True, so this gets colored in console
            return f"[{style}]{prefix}{msg}[/]"
        return f"{prefix}{msg}"


class RoleLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter that merges role metadata into LogRecord via extra=..."""

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        extra = kwargs.get("extra", {})
        merged = {**self.extra, **extra}
        kwargs["extra"] = merged
        return msg, kwargs


def role_logger(name: str, *, role: str, **extra: Any) -> RoleLoggerAdapter:
    """
    Create a logger adapter that always injects a role.

    Usage:
        agent_log = role_logger(__name__, role="agent", agent_id="A1")
        agent_log.info("Hello")
    """
    base = logging.getLogger(name)
    return RoleLoggerAdapter(base, {"role": role, **extra})


def _coerce_level(level: str | int | None, default: int = logging.INFO) -> int:
    if level is None:
        return default
    if isinstance(level, int):
        return level
    level_str = str(level).strip().upper()
    return getattr(logging, level_str, default)


def configure_logging(
    *,
    level: str | int | None = None,
    rich: bool | None = None,
    show_time: bool = True,
    show_path: bool = False,
    show_role: bool = True,
    show_logger: bool = False,
    log_file: str | None = None,
    file_level: str | int | None = None,
) -> None:
    """
    Configure logging for the application.

    Console:
      - RichHandler + RoleFormatter(use_markup=True) if rich=True and Rich is available
      - otherwise StreamHandler (plain)

    File (optional):
      - FileHandler + RoleFormatter(use_markup=False) to avoid markup junk in files

    Env vars:
      - RNB_LOG_LEVEL (e.g., DEBUG/INFO)
      - RNB_LOG_RICH (0/1)
      - RNB_LOG_FILE (path) (optional)
    """
    # Defaults from env vars
    level = _coerce_level(level or os.getenv("RNB_LOG_LEVEL"), default=logging.INFO)

    if rich is None:
        env = os.getenv("RNB_LOG_RICH", "1").strip().lower()
        rich = env not in {"0", "false", "no", "off"}

    if log_file is None:
        log_file = os.getenv("RNB_LOG_FILE")

    file_level = _coerce_level(file_level, default=level)

    root = logging.getLogger()
    root.setLevel(min(level, file_level))

    # Reset handlers to avoid duplicates (notebooks / repeated calls)
    for h in list(root.handlers):
        root.removeHandler(h)

    # ---- Console handler ----
    if rich:
        try:
            from rich.logging import RichHandler  # type: ignore

            console_handler: logging.Handler = RichHandler(
                rich_tracebacks=True,
                show_time=show_time,
                show_path=show_path,
                markup=True,  # required for RoleFormatter markup
            )
            console_formatter: logging.Formatter = RoleFormatter(
                show_role=show_role,
                show_logger=show_logger,
                use_markup=True,
            )
        except Exception:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
    else:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    # ---- File handler (no markup) ----
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)

        # IMPORTANT: no markup in files (plain text)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # If you still want role in file logs, you can swap to:
        # file_formatter = RoleFormatter(show_role=True, show_logger=True, use_markup=False)
        file_handler.setFormatter(file_formatter)

        root.addHandler(file_handler)

    # Silence very chatty libs by default
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Convenience helper. Prefer logging.getLogger(__name__) in modules."""
    return logging.getLogger(name or "rnb")
