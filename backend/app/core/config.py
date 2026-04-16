from __future__ import annotations

"""Application settings loaded from environment variables and optional .env files."""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_list(value: Union[str, Sequence[str], None]) -> List[str]:
    """
    Accepts:
      - None -> []
      - "a,b,c" -> ["a","b","c"]
      - '["a","b"]' (best-effort) -> ["a","b"]
      - ["a","b"] -> ["a","b"]
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        # Already structured input, just normalize whitespace and cast to strings.
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    # Handle a simple JSON-ish list string without adding a new dependency
    if s.startswith("[") and s.endswith("]"):
        s_inner = s[1:-1].strip()
        if not s_inner:
            return []
        # split on commas and strip quotes/spaces
        parts = [p.strip().strip('"').strip("'") for p in s_inner.split(",")]
        return [p for p in parts if p]
    # Comma-separated
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and optional .env file.

    Defaults are chosen to work out-of-the-box for local development, while allowing
    production overrides via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App identity and environment mode.
    app_name: str = Field(default="CUSTOMER CHURN PREDICTION SYSTEM - Backend", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")  # development|staging|production
    debug: bool = Field(default=False, alias="DEBUG")

    # Network binding for the FastAPI/Uvicorn process.
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # Logging controls for console output.
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")

    # CORS controls what frontend origins/browsers may call this backend.
    cors_allow_origins: List[str] = Field(default_factory=list, alias="CORS_ALLOW_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, alias="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default_factory=lambda: ["*"], alias="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default_factory=lambda: ["*"], alias="CORS_ALLOW_HEADERS")
    cors_expose_headers: List[str] = Field(default_factory=list, alias="CORS_EXPOSE_HEADERS")
    cors_max_age: int = Field(default=600, alias="CORS_MAX_AGE")

    # Upload limits (basic guardrail; detailed enforcement happens while streaming to disk).
    max_upload_mb: int = Field(default=25, alias="MAX_UPLOAD_MB")

    # Runtime storage directories for uploads, processed data, model artifacts, and metadata.
    uploads_dir: Path = Field(default=Path("uploads"), alias="UPLOADS_DIR")
    processed_dir: Path = Field(default=Path("processed"), alias="PROCESSED_DIR")
    models_dir: Path = Field(default=Path("models"), alias="MODELS_DIR")
    metadata_dir: Path = Field(default=Path("metadata"), alias="METADATA_DIR")

    # ML defaults used by the training pipeline.
    random_seed: int = Field(default=42, alias="RANDOM_SEED")
    test_size: float = Field(default=0.2, alias="TEST_SIZE")

    # LLM (Ollama) settings for optional local explanation generation.
    llm_enabled: bool = Field(default=True, alias="LLM_ENABLED")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", alias="OLLAMA_MODEL")
    ollama_timeout_s: int = Field(default=60000, alias="OLLAMA_TIMEOUT_S")
    ollama_seed: int = Field(default=42, alias="OLLAMA_SEED")
    ollama_max_tokens: int = Field(default=256, alias="OLLAMA_MAX_TOKENS")

    @field_validator(
        "cors_allow_origins",
        "cors_allow_methods",
        "cors_allow_headers",
        "cors_expose_headers",
        mode="before",
    )
    @classmethod
    def _validate_lists(cls, v):
        return _parse_list(v)

    @field_validator("uploads_dir", "processed_dir", "models_dir", "metadata_dir", mode="before")
    @classmethod
    def _validate_paths(cls, v):
        if isinstance(v, Path):
            return v
        return Path(str(v))

    @field_validator("uploads_dir", "processed_dir", "models_dir", "metadata_dir", mode="after")
    @classmethod
    def _make_paths_absolute_under_backend(cls, v: Path) -> Path:
        # Make relative paths resolve under backend/ (repository backend root).
        # backend/app/core/config.py -> parents[2] = backend/
        backend_root = Path(__file__).resolve().parents[2]
        return v if v.is_absolute() else (backend_root / v).resolve()

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, v):
        return str(v).upper().strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Cache settings so every import site shares one validated config object.
    return Settings()
