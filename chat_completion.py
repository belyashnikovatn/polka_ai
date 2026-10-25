import logging
import os
import httpx
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
# Получаем ключ API
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Создаем клиент OpenAI
openrouter_client = OpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Проверяем, что ключ API установлен
if not openrouter_api_key:
    raise RuntimeError(
        "Переменная окружения OPENROUTER_API_KEY не задана"
    )

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def log_response_usage(response: httpx.Response):
    """Логирование использования токенов."""
    usage = response.usage
    logger.info(
        "Потреблено токенов: %d входных, %d выходных",
        usage.prompt_tokens,
        usage.completion_tokens,
    )


def calculate_retry_delay(attempt: int, base_delay: float = 1.0) -> float:
    """Вычисление задержки для повторного запроса."""
    return base_delay * (2 ** (attempt - 1))


def create_chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_tokens: int = 220,
    # top_k: int = 0,
    # repetition_penalty: float = 1.0,
):
    response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                # top_k=top_k,
                # repetition_penalty=repetition_penalty,
            )

    log_response_usage(response)

    return response


print(create_chat_completion(
    openrouter_client, 
    "openrouter/free", 
    [{"role": "user", "content": "Hello, world!"}]))
