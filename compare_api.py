"""Сравнение локального vLLM и бесплатной модели OpenRouter на одном промпте."""

import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger("compare_api")

# Тот же промпт, что у локального прогона в prompt_test.ipynb.
SYSTEM_PROMPT = (
    "Ты заполняешь карточку товара для витрины интернет-магазина.\n"
    "Верни один JSON-объект и ничего больше. Не используй Markdown и не оборачивай ответ в блок кода. "
    "Первый символ ответа — {. Последний — }. Не пиши ``` и не пиши слово json до объекта.\n\n"
    "ТРЕБОВАНИЯ К КОНТЕНТУ:\n"
    "- product_id: должен строго совпадать с входящим.\n"
    "- description: связный текст от 2 до 5 предложений.\n"
    "- pros: массив преимуществ на основе характеристик и отзывов (от 2 до 5 строк).\n"
    "- cons: массив недостатков на основе отзывов (от 1 до 3 строк).\n"
    "- tags: массив ключевых слов для поиска (от 3 до 8 строк)."
)

CARD_LIMIT = 20
DROPPED: set[str] = set()
MAX_TOKENS = 500
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 20
REPETITION_PENALTY = 1.2
MAX_RATE_RETRIES = 4
MAX_SERVER_RETRIES = 3
MAX_FIXES = 2
INPUT_RUB_PER_MILLION = 20
OUTPUT_RUB_PER_MILLION = 60
REPORT_PATH = Path("outputs/report.json")
API_OUTPUT_PATH = Path("outputs/api_dev.jsonl")

# Локальный прогон тех же 20 карточек dev.jsonl, выбранный набор параметров.
# Вход: среднее число токенов подогнанного промпта (токенайзер Qwen).
# Выход и доля валидных: совместный прогон выбранного набора, 286.4 токена, 18/20.
# Время: средняя задержка одного ответа на этих 20 карточках, 11.37 с.
LOCAL = {
    "label": "Локальная модель на vLLM",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "cards": 20,
    "valid": 18,
    "prompt_tokens": 487.8,
    "completion_tokens": 286.4,
    "latency_seconds": 11.37,
}


REQUIRED_FIELDS = ("product_id", "description", "pros", "cons", "tags")
DESCRIPTION_MIN_CHARS = 120
DESCRIPTION_MAX_CHARS = 700
DESCRIPTION_MIN_SENTENCES = 2
DESCRIPTION_MAX_SENTENCES = 5
LIST_LIMITS = {"pros": (2, 5), "cons": (1, 3), "tags": (3, 8)}
_REVIEW_LEAD_INS = (
    "что могу сказать:",
    "коротко:",
    "в целом",
    "пользуюсь пару недель.",
    "пользуюсь месяц.",
    "пользуюсь каждый день.",
    "купил на замену старому.",
    "брал для себя.",
    "заказывал на подарок.",
)


class CardValidationError(ValueError):
    """Ответ не прошёл проверку формата карточки."""


def count_sentences(text: str) -> int:
    """Число предложений. Точка внутри числа концом предложения не считается."""
    return len([part for part in re.split(r"[.!?]+(?!\d)", text) if part.strip()])


def _string_list(value: object, name: str, minimum: int, maximum: int) -> list[str]:
    if not isinstance(value, list):
        raise CardValidationError(f"Поле {name} должно быть списком")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise CardValidationError(f"Элементы {name} должны быть строками")
        stripped = item.strip()
        if not stripped:
            raise CardValidationError(f"Элементы {name} не должны быть пустыми")
        items.append(stripped)
    if not minimum <= len(items) <= maximum:
        raise CardValidationError(
            f"Поле {name} должно содержать от {minimum} до {maximum} "
            f"пунктов, получено {len(items)}"
        )
    return items


def validate_card(data: object, product_id: str) -> dict:
    """Проверить JSON карточки по правилам витрины."""
    if not isinstance(data, dict):
        raise CardValidationError("На верхнем уровне должен находиться JSON-объект")
    received = set(data)
    missing = [field for field in REQUIRED_FIELDS if field not in received]
    unexpected = sorted(received - set(REQUIRED_FIELDS))
    if missing:
        raise CardValidationError(f"Отсутствуют обязательные поля: {missing}")
    if unexpected:
        raise CardValidationError(f"Получены лишние поля: {unexpected}")
    received_id = data["product_id"]
    if not isinstance(received_id, str):
        raise CardValidationError("Поле product_id должно быть строкой")
    if received_id != product_id:
        raise CardValidationError(
            f"product_id должен быть {product_id}, получено {received_id}"
        )
    description = data["description"]
    if not isinstance(description, str):
        raise CardValidationError("Поле description должно быть строкой")
    description = description.strip()
    description_length = len(description)
    if not DESCRIPTION_MIN_CHARS <= description_length <= DESCRIPTION_MAX_CHARS:
        raise CardValidationError(
            "Поле description должно содержать от "
            f"{DESCRIPTION_MIN_CHARS} до {DESCRIPTION_MAX_CHARS} символов, "
            f"получено {description_length}"
        )
    sentence_count = count_sentences(description)
    if not DESCRIPTION_MIN_SENTENCES <= sentence_count <= DESCRIPTION_MAX_SENTENCES:
        raise CardValidationError(
            "Поле description должно содержать от "
            f"{DESCRIPTION_MIN_SENTENCES} до {DESCRIPTION_MAX_SENTENCES} "
            f"предложений, получено {sentence_count}"
        )
    card = {"product_id": received_id, "description": description}
    for name, (minimum, maximum) in LIST_LIMITS.items():
        card[name] = _string_list(data[name], name, minimum, maximum)
    return card


def review_core(text: str) -> str:
    """Смысл отзыва без повторяющегося зачина."""
    normalized = " ".join(text.lower().split())
    changed = True
    while changed:
        changed = False
        for lead in _REVIEW_LEAD_INS:
            if normalized.startswith(lead):
                normalized = normalized[len(lead):].strip(" :.-")
                changed = True
                break
    return normalized


def unique_reviews(reviews: list) -> list[str]:
    """Один отзыв на смысл. Из дублей остаётся самая короткая формулировка."""
    best: dict[str, str] = {}
    order: list[str] = []
    for review in reviews:
        text = str(review).strip()
        if not text:
            continue
        key = review_core(text)
        if not key:
            continue
        current = best.get(key)
        if current is None:
            best[key] = text
            order.append(key)
        elif len(text) < len(current):
            best[key] = text
    return [best[key] for key in order]


def prepare_product(card: dict) -> dict:
    """Убрать повторяющиеся отзывы перед запросом."""
    prepared = dict(card)
    prepared["reviews"] = unique_reviews(card.get("reviews") or [])
    return prepared


def load_cards(path: str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def require_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Переменная окружения OPENROUTER_API_KEY не задана")
    return api_key


def free_model_ids(api_key: str) -> list[str]:
    response = httpx.get(
        "https://openrouter.ai/api/v1/models",
        params={"sort": "most-popular"},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    response.raise_for_status()
    model_ids = []
    for model in response.json()["data"]:
        architecture = model.get("architecture") or {}
        modalities = architecture.get("input_modalities") or []
        model_id = str(model.get("id", ""))
        if model_id.endswith(":free") and modalities == ["text"]:
            model_ids.append(model_id)
    # Эта модель ответила на пробный запрос. Мёртвые upstream идут в конец.
    preferred = "nvidia/nemotron-3.5-lightning:free"
    if preferred in model_ids:
        model_ids.remove(preferred)
        model_ids.insert(0, preferred)
    return model_ids


def model_is_alive(client: OpenAI, model_id: str) -> bool:
    try:
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Ответь одним словом: ок"}],
            max_tokens=8,
            temperature=0,
        )
    except Exception as error:
        logger.warning("Модель %s недоступна: %s", model_id, error)
        return False
    return True


def select_free_model(client: OpenAI, api_key: str) -> str:
    preferred = os.getenv("OPENROUTER_MODEL")
    if preferred:
        return preferred

    for model_id in free_model_ids(api_key):
        if model_is_alive(client, model_id):
            return model_id
    raise RuntimeError("Ни одна бесплатная текстовая модель не ответила")


def retry_delay(attempt: int, error: Exception | None = None) -> float:
    body = getattr(error, "body", None) or {}
    headers = ((body.get("error") or {}).get("metadata") or {}).get("headers") or {}
    reset = headers.get("X-RateLimit-Reset")
    if reset:
        wait = int(reset) / 1000 - time.time()
        if 0 < wait < 90:
            return wait + 0.5
    return float(2 ** (attempt - 1))


def loads_model_json(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise CardValidationError("На верхнем уровне должен находиться JSON-объект")
    return data


def base_request(model_id: str, card: dict) -> dict:
    prepared = prepare_product(card)
    request = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prepared, ensure_ascii=False)},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "extra_body": {
            "reasoning": {"enabled": False},
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
        },
    }
    if "sampling" in DROPPED:
        request["extra_body"].pop("top_k", None)
        request["extra_body"].pop("repetition_penalty", None)
    if "reasoning" in DROPPED:
        request["extra_body"].pop("reasoning", None)
    if "extra_body" in DROPPED or not request["extra_body"]:
        request.pop("extra_body", None)
    if "response_format" in DROPPED:
        request.pop("response_format", None)
    if "max_tokens" in DROPPED:
        request["max_tokens"] = 256
    return request


def fix_request(request: dict, error: BadRequestError) -> dict | None:
    """Убрать из запроса то, из-за чего сервер ответил 400."""
    message = str(error).lower()
    fixed = dict(request)
    extra = dict(fixed.get("extra_body") or {})
    if extra.get("top_k") is not None or extra.get("repetition_penalty") is not None:
        extra.pop("top_k", None)
        extra.pop("repetition_penalty", None)
        DROPPED.add("sampling")
        if extra:
            fixed["extra_body"] = extra
        else:
            fixed.pop("extra_body", None)
        return fixed
    if extra.get("reasoning") is not None and "reasoning" in message:
        extra.pop("reasoning", None)
        DROPPED.add("reasoning")
        if extra:
            fixed["extra_body"] = extra
        else:
            fixed.pop("extra_body", None)
        return fixed
    if "response_format" in fixed and any(
        word in message for word in ("response_format", "json", "unsupported", "schema", "provider")
    ):
        fixed.pop("response_format", None)
        DROPPED.add("response_format")
        return fixed
    if "maximum context" in message or "too long" in message or "max_tokens" in message:
        if fixed.get("max_tokens", 0) > 256:
            fixed["max_tokens"] = 256
            DROPPED.add("max_tokens")
            return fixed
    return None


def create_completion(client: OpenAI, request: dict):
    """Один запрос с повторами: 429 и 5xx ждут, 400 меняет тело запроса."""
    rate_attempts = 0
    server_attempts = 0
    fixes = 0
    while True:
        try:
            return client.chat.completions.create(**request)
        except BadRequestError as error:
            fixes += 1
            if fixes > MAX_FIXES:
                raise
            fixed = fix_request(request, error)
            if fixed is None:
                raise
            logger.warning("400, запрос изменён: %s", error)
            request.clear()
            request.update(fixed)
        except RateLimitError as error:
            rate_attempts += 1
            if rate_attempts > MAX_RATE_RETRIES:
                raise
            delay = retry_delay(rate_attempts, error)
            logger.warning("429, пауза %.0f с", delay)
            time.sleep(delay)
        except (APITimeoutError, APIConnectionError) as error:
            server_attempts += 1
            if server_attempts > MAX_SERVER_RETRIES:
                raise
            delay = retry_delay(server_attempts)
            logger.warning("сеть, пауза %.0f с: %s", delay, error)
            time.sleep(delay)
        except APIStatusError as error:
            if error.status_code < 500:
                raise
            server_attempts += 1
            if server_attempts > MAX_SERVER_RETRIES:
                raise
            delay = retry_delay(server_attempts)
            logger.warning("%s, пауза %.0f с", error.status_code, delay)
            time.sleep(delay)


def generate_one(client: OpenAI, model_id: str, card: dict) -> dict:
    product_id = card["product_id"]
    started = time.perf_counter()
    result = {
        "product_id": product_id,
        "valid": False,
        "prompt_tokens": None,
        "completion_tokens": None,
        "latency_seconds": None,
        "error": None,
        "card": None,
    }
    try:
        response = create_completion(client, base_request(model_id, card))
    except Exception as error:
        result["error"] = str(error)
        result["latency_seconds"] = time.perf_counter() - started
        return result

    result["latency_seconds"] = time.perf_counter() - started
    usage = response.usage
    if usage is not None:
        result["prompt_tokens"] = usage.prompt_tokens
        result["completion_tokens"] = usage.completion_tokens
    content = (response.choices[0].message.content or "").strip()
    try:
        payload = loads_model_json(content)
        result["card"] = validate_card(payload, product_id)
        result["valid"] = True
    except (json.JSONDecodeError, CardValidationError) as error:
        result["error"] = str(error)
    return result


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def cost_per_thousand(prompt_tokens: float, completion_tokens: float) -> float:
    """Цена 1000 карточек по условному прайсу и фактическому расходу токенов."""
    per_card = (
        prompt_tokens * INPUT_RUB_PER_MILLION
        + completion_tokens * OUTPUT_RUB_PER_MILLION
    ) / 1_000_000
    return per_card * 1000


def side_metrics(label: str, model: str, rows: list[dict]) -> dict:
    prompts = [row["prompt_tokens"] for row in rows if row["prompt_tokens"] is not None]
    completions = [
        row["completion_tokens"] for row in rows if row["completion_tokens"] is not None
    ]
    latencies = [row["latency_seconds"] for row in rows if row["latency_seconds"] is not None]
    valid = sum(row["valid"] for row in rows)
    prompt_mean = mean(prompts)
    completion_mean = mean(completions)
    return {
        "label": label,
        "model": model,
        "cards": len(rows),
        "valid": valid,
        "valid_rate": valid / len(rows) if rows else 0,
        "prompt_tokens": prompt_mean,
        "completion_tokens": completion_mean,
        "latency_seconds": mean(latencies),
        "cost_per_thousand": (
            None
            if prompt_mean is None or completion_mean is None
            else cost_per_thousand(prompt_mean, completion_mean)
        ),
    }


def local_metrics() -> dict:
    metrics = side_metrics(
        LOCAL["label"],
        LOCAL["model"],
        [
            {
                "valid": index < LOCAL["valid"],
                "prompt_tokens": LOCAL["prompt_tokens"],
                "completion_tokens": LOCAL["completion_tokens"],
                "latency_seconds": LOCAL["latency_seconds"],
            }
            for index in range(LOCAL["cards"])
        ],
    )
    metrics["valid"] = LOCAL["valid"]
    metrics["valid_rate"] = LOCAL["valid"] / LOCAL["cards"]
    return metrics


def format_rate(metrics: dict) -> str:
    return f"{metrics['valid']}/{metrics['cards']}, {metrics['valid_rate']:.0%}"


def format_tokens(metrics: dict) -> str:
    if metrics["prompt_tokens"] is None or metrics["completion_tokens"] is None:
        return "нет usage"
    return f"{metrics['prompt_tokens']:.0f} / {metrics['completion_tokens']:.0f}"


def format_latency(metrics: dict) -> str:
    if metrics["latency_seconds"] is None:
        return "нет замера"
    return f"{metrics['latency_seconds']:.2f} с"


def format_cost(metrics: dict) -> str:
    if metrics["cost_per_thousand"] is None:
        return "нет расчёта"
    return f"{metrics['cost_per_thousand']:.2f} ₽"


def print_table(local: dict, remote: dict) -> None:
    rows = (
        ("Доля валидных карточек", format_rate(local), format_rate(remote)),
        ("Токенов на карточку, вход/выход", format_tokens(local), format_tokens(remote)),
        ("Время на карточку", format_latency(local), format_latency(remote)),
        ("Стоимость 1000 карточек", format_cost(local), format_cost(remote)),
    )
    headers = ("Что сравниваем", local["label"], f"{remote['label']}: {remote['model']}")
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(3)]
    line = " | ".join(headers[index].ljust(widths[index]) for index in range(3))
    print(line)
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[index].ljust(widths[index]) for index in range(3)))


def main() -> None:
    api_key = require_api_key()
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=0,
        timeout=90.0,
    )
    model_id = select_free_model(client, api_key)
    cards = load_cards("dev.jsonl")[:CARD_LIMIT]
    logger.info("Модель: %s, карточек: %d", model_id, len(cards))

    results = []
    for index, card in enumerate(cards, start=1):
        result = generate_one(client, model_id, card)
        results.append(result)
        state = "ok" if result["valid"] else "fail"
        logger.info(
            "%d/%d %s %s tokens=%s/%s latency=%.2f %s",
            index,
            len(cards),
            result["product_id"],
            state,
            result["prompt_tokens"],
            result["completion_tokens"],
            result["latency_seconds"] or 0,
            result["error"] or "",
        )

    remote = side_metrics("Модель через API", model_id, results)
    local = local_metrics()
    print()
    print_table(local, remote)

    API_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with API_OUTPUT_PATH.open("w", encoding="utf-8") as sink:
        for result in results:
            if result["card"] is not None:
                sink.write(json.dumps(result["card"], ensure_ascii=False) + "\n")

    report = {
        "price_rub_per_million": {
            "input": INPUT_RUB_PER_MILLION,
            "output": OUTPUT_RUB_PER_MILLION,
        },
        "local": local,
        "api": remote,
        "api_cards": [
            {key: value for key, value in result.items() if key != "card"}
            for result in results
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Отчёт: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
