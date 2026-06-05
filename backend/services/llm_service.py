from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from config import get_settings


_llm_instance: OpenAILike | None = None


def get_llm() -> OpenAILike:
    global _llm_instance
    if _llm_instance is None:
        settings = get_settings()
        _llm_instance = OpenAILike(
            model=settings.deepseek_model,
            api_base=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
            is_chat_model=True,
            timeout=120.0,
        )
        Settings.llm = _llm_instance
    return _llm_instance
