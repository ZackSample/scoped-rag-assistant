from concurrent.futures import ThreadPoolExecutor

from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.rag.openai_client import embedding_model_name, get_client


@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(4))
def _embed_batch(
    client: OpenAI | AzureOpenAI, model: str, texts: list[str], dimensions: int | None
) -> list[list[float]]:
    # Retry transient provider failures so one network hiccup does not kill indexing.
    kwargs = {"model": model, "input": texts}
    if dimensions:
        # text-embedding-3-* models can truncate their native output width
        # while keeping most of the quality.
        kwargs["dimensions"] = dimensions
    response = client.embeddings.create(**kwargs)
    return [item.embedding for item in response.data]


def _embedding_client() -> tuple[OpenAI | AzureOpenAI, str, int | None]:
    """Return the client, model, and dimension override for embeddings.

    When EMBEDDING_BASE_URL is set, embeddings go to a local OpenAI-compatible
    server while chat and the judge stay on whatever provider is configured.
    The `dimensions` parameter is not sent in that case - local servers reject
    it - so the model's native width is used as-is.
    """

    settings = get_settings()
    if settings.embedding_base_url:
        client = OpenAI(
            api_key=settings.embedding_api_key or "not-needed",
            base_url=settings.embedding_base_url,
            timeout=settings.openai_timeout_seconds,
            max_retries=2,
        )
        return client, settings.embedding_model, None
    return get_client(), embedding_model_name(), settings.embedding_dimensions


def _build_batches(texts: list[str], max_items: int, max_chars: int) -> list[list[str]]:
    """Split texts into requests bounded by BOTH item count and size.

    Batching purely by item count ignores the provider's per-request token cap:
    a spare-parts list is mostly part numbers, which tokenise closer to 2
    characters per token instead of prose's ~4. A batch of dense chunks can
    blow the cap while a batch of the same number of prose chunks fits easily,
    and the whole file then fails with a 400 though no single chunk looks wrong.

    Characters are the proxy since counting real tokens would mean shipping a
    tokeniser. A single text longer than the budget still gets its own request.
    """

    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for text in texts:
        length = len(text)
        if current and (len(current) >= max_items or current_chars + length > max_chars):
            batches.append(current)
            current, current_chars = [], 0
        current.append(text)
        current_chars += length
    if current:
        batches.append(current)
    return batches


def embed_texts(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """Embed texts, preserving input order. Batches are sent concurrently."""

    # Empty strings are not valid embedding inputs, so send a harmless space.
    cleaned = [text if text.strip() else " " for text in texts]
    if not cleaned:
        return []

    settings = get_settings()
    client, model, dimensions = _embedding_client()
    size = batch_size or settings.embedding_batch_size
    batches = _build_batches(cleaned, size, settings.embedding_max_chars_per_request)

    if len(batches) == 1 or settings.embedding_concurrency <= 1:
        return [vector for batch in batches for vector in _embed_batch(client, model, batch, dimensions)]

    # map() keeps results in submission order, so vectors still line up with
    # the input texts - callers zip them straight onto their chunks.
    with ThreadPoolExecutor(max_workers=settings.embedding_concurrency) as pool:
        results = pool.map(lambda batch: _embed_batch(client, model, batch, dimensions), batches)
        return [vector for batch_result in results for vector in batch_result]
