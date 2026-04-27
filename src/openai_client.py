from __future__ import annotations

import json
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime for local-only mode
    OpenAI = None  # type: ignore[assignment]


class OpenAIJsonClient:
    """Small wrapper that keeps OpenAI usage optional and JSON-focused."""

    def __init__(self, api_key: str | None, model: str):
        self.api_key = (api_key or "").strip()
        self.model = model.strip() or "gpt-5.4-mini"
        self.enabled = bool(self.api_key) and OpenAI is not None
        self._client = OpenAI(api_key=self.api_key) if self.enabled else None

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> dict[str, Any] | None:
        if not self.enabled or self._client is None:
            return None

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"_error": "Model did not return a JSON object."}
        except Exception as exc:  # pragma: no cover - depends on external service
            return {"_error": str(exc)}
