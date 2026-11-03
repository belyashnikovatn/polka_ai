"""Пакетный прогон benchmark.jsonl на локальном vLLM.

Та же схема, что финальная ячейка бенчмарка в prompt_test.ipynb:
один запрос на карточку, concurrency 32, выбранные параметры,
карточка пишется одной JSON-строкой. Сервер должен уже слушать порт 8000.
После прогона секции run и metrics в outputs/report.json обновляются,
api_comparison остаётся как есть.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from check_metrics import merge_benchmark, quality_from_records, validate_card

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_URL = "http://localhost:8000/v1"
BENCHMARK_PATH = Path("benchmark.jsonl")
OUTPUT_PATH = Path("outputs/predictions.jsonl")
PARTIAL_PATH = Path("outputs/predictions.partial.jsonl")
REPORT_PATH = Path("outputs/report.json")
CONCURRENCY = 32
MAX_INPUT_TOKENS = 1500
CARD_FIELDS = ("product_id", "description", "pros", "cons", "tags")
CHOSEN = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 20,
    "repetition_penalty": 1.2,
    "max_tokens": 500,
}
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
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "product_id": {"type": "string"},
        "description": {"type": "string", "minLength": 120, "maxLength": 700},
        "pros": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5},
        "cons": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 8},
    },
    "required": ["product_id", "description", "pros", "cons", "tags"],
    "additionalProperties": False,
}
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


def review_core(text: str) -> str:
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


def spread_reviews(items: list[str], count: int) -> list[str]:
    """Равномерно взять count отзывов, включая первый и последний."""
    if count >= len(items):
        return list(items)
    if count <= 0:
        return []
    if count == 1:
        return [items[0]]
    last = len(items) - 1
    picked = []
    seen = set()
    for index in range(count):
        position = round(index * last / (count - 1))
        if position not in seen:
            seen.add(position)
            picked.append(items[position])
    return picked


def fit_product(card: dict, token_count, max_tokens: int = MAX_INPUT_TOKENS) -> dict:
    """Оставить уникальные отзывы и уложиться в max_tokens на весь промпт."""

    def count_reviews(reviews: list[str]) -> int:
        return token_count({**card, "reviews": reviews})

    unique = unique_reviews(card.get("reviews") or [])
    if not unique or count_reviews(unique) <= max_tokens:
        selected = unique
    else:
        selected = []
        low, high = 0, len(unique)
        while low <= high:
            mid = (low + high) // 2
            candidate = spread_reviews(unique, mid)
            if count_reviews(candidate) <= max_tokens:
                selected = candidate
                low = mid + 1
            else:
                high = mid - 1
    fitted = dict(card)
    fitted["reviews"] = selected
    return fitted


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
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("На верхнем уровне должен находиться объект", text, 0)
    return payload


def card_record(payload: object, product_id: str) -> dict:
    if not isinstance(payload, dict):
        return {"product_id": product_id}
    record = {name: payload.get(name) for name in CARD_FIELDS}
    if not record["product_id"]:
        record["product_id"] = product_id
    return record


def server_is_up(base_url: str) -> bool:
    try:
        response = httpx.get(base_url.rstrip("/") + "/models", timeout=5.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def load_benchmark(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def make_token_counter():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def count_prompt_tokens(product: dict) -> int:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(product, ensure_ascii=False)},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        return len(encoded)

    return count_prompt_tokens


async def run_concurrent(items: list[dict], concurrency: int, generate_one):
    semaphore = asyncio.Semaphore(concurrency)

    async def wrapped(item: dict) -> dict:
        async with semaphore:
            return await generate_one(item)

    started = time.perf_counter()
    results = await asyncio.gather(*[wrapped(item) for item in items])
    return results, time.perf_counter() - started


def model_answered(item: dict) -> bool:
    """Ответ модели дошёл. Обрыв связи и таймаут сюда не входят."""
    if item["completion_tokens"] is not None or item["valid"]:
        return True
    error = (item["error"] or "").lower()
    transport = ("connection", "connect", "timed out", "timeout", "unavailable")
    return bool(error) and not any(marker in error for marker in transport)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


async def generate_all(items: list[dict], concurrency: int) -> tuple[list[dict], float]:
    client = AsyncOpenAI(base_url=BASE_URL, api_key="local-vllm")
    write_lock = asyncio.Lock()
    finished = 0
    PARTIAL_PATH.write_text("", encoding="utf-8")

    async def generate_one(card: dict) -> dict:
        nonlocal finished
        started = time.perf_counter()
        result = {
            "product_id": card["product_id"],
            "valid": False,
            "record": {"product_id": card["product_id"]},
            "completion_tokens": None,
            "latency": None,
            "error": None,
        }
        extra_body = {
            "structured_outputs": {"json": JSON_SCHEMA},
            "repetition_penalty": CHOSEN["repetition_penalty"],
            "top_k": CHOSEN["top_k"],
        }
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(card, ensure_ascii=False)},
                ],
                temperature=CHOSEN["temperature"],
                top_p=CHOSEN["top_p"],
                max_tokens=CHOSEN["max_tokens"],
                extra_body=extra_body,
                timeout=180,
            )
        except Exception as error:
            result["error"] = str(error)
            result["latency"] = time.perf_counter() - started
        else:
            usage = response.usage
            result["completion_tokens"] = usage.completion_tokens if usage is not None else None
            result["latency"] = time.perf_counter() - started
            content = (response.choices[0].message.content or "").strip()
            try:
                payload = loads_model_json(content)
            except json.JSONDecodeError as error:
                result["error"] = str(error)
            else:
                result["record"] = card_record(payload, card["product_id"])
                errors = validate_card(payload, card["product_id"])
                if errors:
                    result["error"] = "; ".join(errors)
                else:
                    result["valid"] = True

        async with write_lock:
            finished += 1
            with PARTIAL_PATH.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(result["record"], ensure_ascii=False) + "\n")
            if finished % 20 == 0 or finished == len(items):
                print(f"  {finished}/{len(items)}", flush=True)
            if result["error"]:
                print(f"  {card['product_id']} fail: {result['error']}", flush=True)
        return result

    try:
        return await run_concurrent(items, concurrency, generate_one)
    finally:
        await client.close()


def main() -> int:
    if not BENCHMARK_PATH.exists():
        print(f"Нет файла {BENCHMARK_PATH}", file=sys.stderr)
        return 1
    if not server_is_up(BASE_URL):
        print(
            f"vLLM не отвечает на {BASE_URL}/models. "
            "predictions.jsonl и report.json не изменены.",
            file=sys.stderr,
        )
        return 1

    source = load_benchmark(BENCHMARK_PATH)
    count_prompt_tokens = make_token_counter()
    items = []
    input_tokens = 0
    for card in source:
        fitted = fit_product(card, count_prompt_tokens)
        input_tokens += count_prompt_tokens(fitted)
        items.append(fitted)
    print(f"Карточек: {len(items)}, одновременно: {CONCURRENCY}")

    results, elapsed = asyncio.run(generate_all(items, CONCURRENCY))
    records = [item["record"] for item in results]
    write_jsonl(OUTPUT_PATH, records)

    answered = [item for item in results if model_answered(item)]
    latencies = [item["latency"] for item in results if item["latency"] is not None]
    completions = [
        item["completion_tokens"] for item in results if item["completion_tokens"] is not None
    ]
    quality = quality_from_records(records, [card["product_id"] for card in source])
    print(
        f"Валидных: {quality['valid']}/{quality['n']}. "
        f"Время: {elapsed:.1f} с. "
        f"Скорость: {quality['n'] / elapsed:.2f} карточек/с. "
        f"Средняя задержка: {sum(latencies) / len(latencies):.2f} с. "
        f"Карточки: {OUTPUT_PATH}"
    )

    if len(answered) != len(results):
        print("Часть запросов не дошла до модели. report.json не обновлён.", file=sys.stderr)
        return 1

    run = {
        "model": MODEL,
        "concurrency": CONCURRENCY,
        "params": CHOSEN,
        "elapsed_sec": round(elapsed, 1),
        "input_tokens": input_tokens,
        "n": len(results),
        "avg_latency": round(sum(latencies) / len(latencies), 2),
        "avg_attempts": 1.0,
    }
    if completions:
        run["output_tokens"] = int(sum(completions))
    merge_benchmark(REPORT_PATH, run, quality)
    print(f"Отчёт: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
