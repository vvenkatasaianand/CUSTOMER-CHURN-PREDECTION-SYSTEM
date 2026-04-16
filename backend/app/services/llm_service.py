from __future__ import annotations

"""Very small wrapper around Ollama's generate API with JSON-only responses."""

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.core.config import Settings
from app.core.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class LLMService:
    settings: Settings

    def is_enabled(self) -> bool:
        # Feature flag allows the rest of the backend to keep working without Ollama.
        return bool(self.settings.llm_enabled)

    def model_name(self) -> str:
        return str(self.settings.ollama_model)

    def generate_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Call Ollama generate endpoint with JSON-only output.
        Returns parsed JSON dict or None on failure.
        """
        if not self.is_enabled():
            logger.info("llm_disabled")
            return None

        # Force deterministic-ish JSON output so calling services can validate and trust the shape.
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "seed": int(self.settings.ollama_seed),
                "num_predict": int(self.settings.ollama_max_tokens),
            },
        }

        url = self.settings.ollama_base_url.rstrip("/") + "/api/generate"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=self.settings.ollama_timeout_s) as resp:
                status_code = getattr(resp, "status", None)
                raw_bytes = resp.read()
                raw = raw_bytes.decode("utf-8")
            logger.info(
                "llm_response_received",
                extra={
                    "status_code": status_code,
                    "response_bytes": len(raw_bytes),
                },
            )
            parsed = json.loads(raw)
            text = parsed.get("response", "")
            if not text:
                logger.warning("llm_empty_response_field")
                return None
            # Ollama returns the model text inside "response"; the project expects that text to itself be JSON.
            return json.loads(text)
        except Exception as exc:
            logger.exception("llm_generate_failed", extra={"error_type": type(exc).__name__})
            return None
