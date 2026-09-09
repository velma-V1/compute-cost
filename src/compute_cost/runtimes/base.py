"""Runtime adapter protocol used by the benchmark runner."""

from __future__ import annotations

from typing import Any, Protocol


class RuntimeAdapter(Protocol):
    def health(self) -> dict[str, Any]: ...

    def version(self) -> dict[str, Any]: ...

    def list_models(self) -> dict[str, Any]: ...

    def model_info(self, model: str) -> dict[str, Any]: ...

    def is_model_available(self, model: str) -> bool: ...

    def generate(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        *,
        stream: bool = True,
        request_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def unload(self, model: str) -> dict[str, Any]: ...

    def pull(self, model: str) -> dict[str, Any]: ...
