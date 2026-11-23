from openai import AzureOpenAI, OpenAI

from app.config import Settings, get_settings


def _is_azure(settings: Settings) -> bool:
    # Azure can be selected explicitly or implied by an endpoint being present.
    return settings.openai_provider.lower() == "azure" or bool(settings.azure_openai_endpoint)


def get_client() -> OpenAI | AzureOpenAI:
    settings = get_settings()
    if _is_azure(settings):
        # Azure OpenAI needs endpoint/version/deployment details, not just a key.
        if not settings.azure_openai_api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is not set.")
        if not settings.azure_openai_endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set.")
        return AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            timeout=settings.openai_timeout_seconds,
            max_retries=2,
        )

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    # OPENAI_BASE_URL, when set in the environment, is picked up by the SDK.
    return OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout_seconds, max_retries=2)


# Reasoning model families only accept the default temperature; passing any
# other value is a 400. The evidence judge therefore cannot be made fully
# deterministic on them.
_NO_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def chat_completion(client: OpenAI | AzureOpenAI, **kwargs):
    """Create a chat completion, adapting arguments to the model's quirks.

    Do NOT add reasoning_effort here. Some endpoints reject it whenever function
    tools are present and point at the Responses API instead, which is what the
    tool loop in app/rag/tools.py uses.
    """

    model = str(kwargs.get("model") or "")
    if model.startswith(_NO_TEMPERATURE_PREFIXES):
        kwargs.pop("temperature", None)
    return client.chat.completions.create(**kwargs)


def chat_model_name() -> str:
    settings = get_settings()
    if _is_azure(settings):
        # Azure calls use deployment names where OpenAI calls use model names.
        return settings.azure_openai_chat_deployment
    return settings.chat_model


def embedding_model_name() -> str:
    settings = get_settings()
    if _is_azure(settings):
        return settings.azure_openai_embedding_deployment
    return settings.embedding_model
