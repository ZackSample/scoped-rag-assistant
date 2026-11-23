from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    # Keep all runtime knobs in one object so local and container runs use the
    # same names instead of hard-coded paths spread through the codebase.

    # --- Model provider -----------------------------------------------------
    # "openai" talks to the OpenAI API (or any server set via OPENAI_BASE_URL);
    # "azure" talks to an Azure OpenAI resource. Setting AZURE_OPENAI_ENDPOINT
    # also selects Azure.
    openai_provider: str = Field(default="openai", alias="OPENAI_PROVIDER")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    # The Responses API (used by the tool loop for streamed reasoning) does not
    # exist on older Azure API versions.
    azure_openai_api_version: str = Field(default="2025-04-01-preview", alias="AZURE_OPENAI_API_VERSION")
    azure_openai_chat_deployment: str = Field(default="", alias="AZURE_OPENAI_CHAT_DEPLOYMENT")
    azure_openai_embedding_deployment: str = Field(default="", alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    chat_model: str = Field(default="gpt-5-mini", alias="CHAT_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    # Optional output width for text-embedding-3-* models, which can truncate
    # their native vectors. Empty means "the model's native width".
    embedding_dimensions: int | None = Field(default=None, alias="EMBEDDING_DIMENSIONS")
    # Point embeddings at a local OpenAI-compatible server (Ollama, vLLM, TEI)
    # while chat stays on the main provider. Deliberately separate from the chat
    # client so a local-embedding experiment changes one variable, not two.
    embedding_base_url: str = Field(default="", alias="EMBEDDING_BASE_URL")
    embedding_api_key: str = Field(default="not-needed", alias="EMBEDDING_API_KEY")
    # Indexing sends one request per batch; without concurrency it waits out a
    # full round trip for each. Kept well below typical rate limits so a large
    # reindex cannot starve the live app of embedding quota.
    embedding_batch_size: int = Field(default=128, alias="EMBEDDING_BATCH_SIZE")
    embedding_concurrency: int = Field(default=4, alias="EMBEDDING_CONCURRENCY")
    # Embedding requests are capped by tokens, not items. Token-dense content
    # (part numbers, spec tables) can exceed that at an item count prose handles
    # fine. 400,000 characters is ~200,000 tokens even at 2 chars/token.
    embedding_max_chars_per_request: int = Field(default=400_000, alias="EMBEDDING_MAX_CHARS_PER_REQUEST")

    # How hard the model thinks before acting, and whether it emits a readable
    # summary of that thinking. Only used by reasoning models through the
    # Responses API. Leave REASONING_EFFORT empty for a non-reasoning model.
    reasoning_effort: str = Field(default="low", alias="REASONING_EFFORT")
    reasoning_summary: str = Field(default="auto", alias="REASONING_SUMMARY")
    openai_timeout_seconds: float = Field(default=60.0, alias="OPENAI_TIMEOUT_SECONDS")

    # --- Documents ----------------------------------------------------------
    # One sub-folder per machine, named "IMM <number> - <make/model> <serial>".
    docs_root: Path = Field(default=Path("data/documents"), alias="DOCS_ROOT")
    # Machine-agnostic documents searched alongside every machine. Defaults to
    # DOCS_ROOT/_general (underscore folders are never treated as machines).
    general_docs_path: Path | None = Field(default=None, alias="GENERAL_DOCS_PATH")
    general_machine_id: str = Field(default="general", alias="GENERAL_MACHINE_ID")
    # Curated identity facts per machine (family, tonnage, fitted screw...),
    # injected into the answering model's context so it can tell which row of a
    # multi-machine table belongs to the machine it is answering about. None
    # means the machine_facts.json that ships next to the machine registry.
    machine_facts_path: Path | None = Field(default=None, alias="MACHINE_FACTS_PATH")

    # --- Storage ------------------------------------------------------------
    # One SQLite file holds the chunk index, users, and chat history.
    app_db_path: Path = Field(default=Path("data/app.db"), alias="APP_DB_PATH")

    # --- Auth ---------------------------------------------------------------
    # Signs the session cookie. MUST be a stable random value in any real
    # deployment: if it changes, every session is invalidated, and if it is
    # guessable, cookies can be forged.
    session_secret: str = Field(default="", alias="SESSION_SECRET")
    # Lets the login gate be switched off for local development without having
    # to create users first. Never set this to false in a deployed environment.
    auth_required: bool = Field(default=True, alias="AUTH_REQUIRED")
    # The session cookie is Secure-only by default. A browser silently discards
    # a Secure cookie over plain http, so signing in on http://localhost needs
    # this set to false.
    session_https_only: bool = Field(default=True, alias="SESSION_HTTPS_ONLY")

    # --- Parsing and retrieval ---------------------------------------------
    # Kill switch for table-aware PDF parsing (see app/rag/parse_pdf.py). Only
    # matters to indexing runs.
    pdf_table_extraction: bool = Field(default=True, alias="PDF_TABLE_EXTRACTION")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    # How deep the ranked vector search looks before diversification trims it.
    # Kept far above what reaches the model on purpose: spec tables (datasheets) rank
    # far below prose that merely discusses the same term, so a shallow pool
    # silently makes them unreachable. See RETRIEVAL_PASSES in rag/answer.py.
    retrieval_candidate_k: int = Field(default=500, alias="RETRIEVAL_CANDIDATE_K")
    retrieval_max_chunks_per_source: int = Field(default=4, alias="RETRIEVAL_MAX_CHUNKS_PER_SOURCE")
    keyword_candidate_k: int = Field(default=60, alias="KEYWORD_CANDIDATE_K")

    # --- Previews -----------------------------------------------------------
    # LibreOffice renders xlsx/pptx citation previews as page images. Optional:
    # without it, previews fall back to the extracted sheet/slide text.
    libreoffice_path: str = Field(default="soffice", alias="LIBREOFFICE_PATH")
    office_preview_cache_dir: Path = Field(default=Path("data/office_preview_cache"), alias="OFFICE_PREVIEW_CACHE_DIR")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def general_root(self) -> Path:
        return self.general_docs_path or self.docs_root / "_general"


@lru_cache
def get_settings() -> Settings:
    # Cache settings so repeated RAG/database calls do not re-read .env.
    return Settings()
