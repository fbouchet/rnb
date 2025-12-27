# src/rnb/console.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .logging import role_logger


@dataclass(frozen=True)
class Console:
    """
    Small convenience layer for role-colored output using logging.

    Example:
        from rnb.logging import configure_logging
        from rnb.console import Console

        configure_logging(level="INFO", rich=True, log_file="rnb.log")
        c = Console(namespace="rnb.demo", agent_id="A1")
        c.user("Hi")
        c.agent("Hello!")
        c.model("Thinking...")
    """

    namespace: str = "rnb"
    agent_id: str | None = None

    def _log(self, role: str):
        extra: dict[str, Any] = {}
        if self.agent_id is not None:
            extra["agent_id"] = self.agent_id
        return role_logger(self.namespace, role=role, **extra)

    # --- role-specific methods ---
    def user(self, message: str, **extra: Any) -> None:
        self._log("user").info(message, extra=extra)

    def agent(self, message: str, **extra: Any) -> None:
        self._log("agent").info(message, extra=extra)

    def model(self, message: str, **extra: Any) -> None:
        self._log("model").info(message, extra=extra)

    def system(self, message: str, **extra: Any) -> None:
        self._log("system").info(message, extra=extra)

    def debug(self, message: str, **extra: Any) -> None:
        self._log("debug").debug(message, extra=extra)

    def warn(self, message: str, **extra: Any) -> None:
        self._log("system").warning(message, extra=extra)

    def error(self, message: str, **extra: Any) -> None:
        self._log("error").error(message, extra=extra)

    # --- formatting helpers (factorized headers/separators) ---
    def blank(self, n: int = 1) -> None:
        """Emit N blank lines via system channel."""
        for _ in range(max(0, n)):
            self.system("")

    def sep(self, *, char: str = "-", width: int = 70) -> None:
        """Horizontal separator."""
        self.system(char * width)

    def rule(
        self,
        title: str | None = None,
        *,
        char: str = "=",
        width: int = 70,
        pad: int = 2,
    ) -> None:
        """
        Header rule.

        If title is provided, prints:
            ========
              title
            ========
        Otherwise prints a single rule line.
        """
        if title:
            self.blank(1)
            self.system(char * width)
            self.system(f"{' ' * pad}{title}")
            self.system(char * width)
            self.blank(1)
        else:
            self.system(char * width)


# -------------------------------------------------------------------
# Convenience functions (for scripts that don't want to instantiate Console)
# -------------------------------------------------------------------


def _c(namespace: str = "rnb", agent_id: str | None = None) -> Console:
    return Console(namespace=namespace, agent_id=agent_id)


def say_user(
    message: str, *, agent_id: str | None = None, namespace: str = "rnb", **extra: Any
) -> None:
    _c(namespace, agent_id).user(message, **extra)


def say_agent(
    message: str, *, agent_id: str | None = None, namespace: str = "rnb", **extra: Any
) -> None:
    _c(namespace, agent_id).agent(message, **extra)


def say_model(
    message: str, *, agent_id: str | None = None, namespace: str = "rnb", **extra: Any
) -> None:
    _c(namespace, agent_id).model(message, **extra)


def say_system(
    message: str, *, agent_id: str | None = None, namespace: str = "rnb", **extra: Any
) -> None:
    _c(namespace, agent_id).system(message, **extra)


def blank(n: int = 1, *, agent_id: str | None = None, namespace: str = "rnb") -> None:
    _c(namespace, agent_id).blank(n)


def sep(
    *,
    char: str = "-",
    width: int = 70,
    agent_id: str | None = None,
    namespace: str = "rnb",
) -> None:
    _c(namespace, agent_id).sep(char=char, width=width)


def rule(
    title: str | None = None,
    *,
    char: str = "=",
    width: int = 70,
    pad: int = 2,
    agent_id: str | None = None,
    namespace: str = "rnb",
) -> None:
    _c(namespace, agent_id).rule(title, char=char, width=width, pad=pad)
