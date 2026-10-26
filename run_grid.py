import json
import logging
import os
import time
from pathlib import Path

import httpx

from cards_exploration import load_cards
from check_metrics import CardValidationError, validate_card
from prompts import SYSTEM_PROMPT, render_product

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger("run_grid")

CARD_LIMIT = 20
API_ATTEMPTS = 3
OUTPUT_PATH = Path("outputs/param_grid.json")
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
# Короткое окно: на 6 ГБ полный контекст Qwen не помещается.
OLLAMA_NUM_CTX = 4096

# Остальные параметры держатся на этих значениях, меняется только один.
BASELINE = {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "max_tokens": 350,
}

SWEEP = {
    "temperature": [0.0, 0.3, 0.7],
    "top_p": [0.9, 1.0],
    "top_k": [20, 50, None],
    "repetition_penalty": [1.0, 1.1, 1.2],
    "max_tokens": [250, 350, 500],
}


def format_value(parameter: str, value: object) -> str:
    if parameter == "top_k" and value is None:
        return "без ограничения"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def config_key(config: dict) -> tuple:
    return (
        config["temperature"],
        config["top_p"],
        config["top_k"],
        config["repetition_penalty"],
        config["max_tokens"],
    )


def build_runs() -> list[dict]:
    """Одна точка сетки на каждое значение параметра. Базовая комбинация считается один раз."""
    runs: list[dict] = []
    seen: dict[tuple, dict] = {}

    for parameter, values in SWEEP.items():
        for value in values:
            config = dict(BASELINE)
            config[parameter] = value
            key = config_key(config)
            run = seen.get(key)
            if run is None:
                run = {"config": config, "rows": []}
                seen[key] = run
                runs.append(run)
            run["rows"].append({"parameter": parameter, "value": value})
    return runs


def extra_body(config: dict) -> dict:
    body = {"repetition_penalty": config["repetition_penalty"]}
    if config["top_k"] is not None:
        body["top_k"] = config["top_k"]
    return body


def has_overlap(items: list[str]) -> bool:
    """Совпадение вроде «комфорт» и «комфортабельность» внутри одного списка."""
    normalized = [item.lower().strip() for item in items]
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            if left and right and (left == right or left in right or right in left):
                return True
    return False


def request_openrouter(model: str, messages: list[dict[str, str]], config: dict):
    from openai import APIError

    from chat_completion import (
        calculate_retry_delay,
        create_chat_completion,
        openrouter_client,
    )

    last_error: APIError | None = None
    for attempt in range(1, API_ATTEMPTS + 1):
        try:
            response = create_chat_completion(
                openrouter_client,
                model,
                messages,
                temperature=config["temperature"],
                top_p=config["top_p"],
                max_tokens=config["max_tokens"],
                response_format={"type": "json_object"},
                extra_body=extra_body(config),
            )
            choice = response.choices[0]
            usage = response.usage
            tokens = usage.completion_tokens if usage is not None else None
            return choice.message.content or "", tokens, choice.finish_reason == "length"
        except APIError as error:
            last_error = error
            logger.warning("api_error attempt=%d/%d %s", attempt, API_ATTEMPTS, error)
            if attempt == API_ATTEMPTS:
                break
            time.sleep(calculate_retry_delay(attempt))
    raise last_error  # type: ignore[misc]


def request_ollama(model: str, messages: list[dict[str, str]], config: dict):
    top_k = config["top_k"] if config["top_k"] is not None else -1
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "top_k": top_k,
                "repeat_penalty": config["repetition_penalty"],
                "num_predict": config["max_tokens"],
                "num_ctx": OLLAMA_NUM_CTX,
            },
        },
        timeout=600.0,
    )
    response.raise_for_status()
    payload = response.json()
    content = (payload.get("message") or {}).get("content") or ""
    return content, payload.get("eval_count"), payload.get("done_reason") == "length"


def evaluate_card(card: dict, model: str, config: dict, backend: str) -> dict:
    """Одна попытка генерации. Повтор только при ошибке API, не при браке формата."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_product(card)},
    ]
    product_id = card["product_id"]
    result = {
        "product_id": product_id,
        "valid": False,
        "description_length": None,
        "completion_tokens": None,
        "truncated": False,
        "tag_overlap": False,
        "error": None,
    }
    try:
        if backend == "ollama":
            content, tokens, truncated = request_ollama(model, messages, config)
        else:
            content, tokens, truncated = request_openrouter(model, messages, config)
    except Exception as error:
        result["error"] = str(error)
        return result

    result["completion_tokens"] = tokens
    result["truncated"] = truncated

    try:
        data = json.loads(content)
        card_json = validate_card(data, product_id)
    except (json.JSONDecodeError, CardValidationError) as error:
        result["error"] = str(error)
        return result

    result["valid"] = True
    result["description_length"] = len(card_json["description"])
    result["tag_overlap"] = has_overlap(card_json["pros"]) or has_overlap(card_json["tags"])
    return result


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize(parameter: str, value: object, outcomes: list[dict]) -> dict:
    valid = [item for item in outcomes if item["valid"]]
    tokens = [
        item["completion_tokens"]
        for item in outcomes
        if item["completion_tokens"] is not None
    ]
    lengths = [item["description_length"] for item in valid]
    return {
        "parameter": parameter,
        "value": value,
        "value_label": format_value(parameter, value),
        "total": len(outcomes),
        "valid_count": len(valid),
        "valid_rate": len(valid) / len(outcomes) if outcomes else 0.0,
        "avg_description_length": mean(lengths),
        "avg_completion_tokens": mean(tokens),
        "truncated": sum(1 for item in outcomes if item["truncated"]),
        "tag_overlap_count": sum(1 for item in valid if item["tag_overlap"]),
    }


def choose_max_tokens(rows: list[dict]) -> dict:
    without_cutoff = [row for row in rows if row["truncated"] == 0]
    if without_cutoff:
        return min(without_cutoff, key=lambda row: row["value"])
    return max(rows, key=lambda row: (row["valid_rate"], row["value"]))


def choose_by_rate(rows: list[dict], preferred: object) -> dict:
    best_rate = max(row["valid_rate"] for row in rows)
    contenders = [row for row in rows if row["valid_rate"] == best_rate]
    for row in contenders:
        if row["value"] == preferred:
            return row
    return contenders[0]


def choose_penalty(rows: list[dict]) -> dict:
    """Сначала доля валидных, при равенстве — меньше повторов в pros и tags."""
    best_rate = max(row["valid_rate"] for row in rows)
    contenders = [row for row in rows if row["valid_rate"] == best_rate]
    return min(
        contenders,
        key=lambda row: (
            row["tag_overlap_count"],
            0 if row["value"] == BASELINE["repetition_penalty"] else 1,
        ),
    )


def justify(table: list[dict], chosen: dict[str, dict]) -> list[str]:
    lines = [
        "Сетка однофакторная: в каждой строке от базы отличается только названный параметр. "
        "База: temperature 0.3, top_p 0.9, top_k 50, repetition_penalty 1.1, max_tokens 350. "
        "На карточку одна попытка, без корректирующего повтора."
    ]
    for parameter, row in chosen.items():
        peers = [item for item in table if item["parameter"] == parameter]
        peer_text = "; ".join(
            f"{item['value_label']}: валидных {item['valid_count']}/{item['total']}, "
            f"длина {format_number(item['avg_description_length'])}, "
            f"токены {format_number(item['avg_completion_tokens'])}, "
            f"обрывов {item['truncated']}, повторов {item['tag_overlap_count']}"
            for item in peers
        )
        if parameter == "max_tokens" and row["truncated"] == 0:
            reason = "наименьшее значение, при котором JSON уже не обрывается"
        elif parameter == "max_tokens":
            reason = "обрывы есть на всех проверенных значениях, взято лучшее по доле валидных"
        elif parameter == "repetition_penalty" and len({item["valid_rate"] for item in peers}) == 1:
            reason = "при той же доле валидных меньше повторов внутри pros и tags"
        else:
            reason = "выше доля валидных карточек"
        lines.append(
            f"{parameter}={row['value_label']}: {reason}. Сравнение: {peer_text}."
        )
    return lines


def format_number(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def print_table(table: list[dict]) -> None:
    header = (
        f"{'параметр':<20} {'значение':<18} {'валидные':>10} "
        f"{'длина':>8} {'токены':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in table:
        valid = f"{row['valid_count']}/{row['total']}"
        print(
            f"{row['parameter']:<20} {row['value_label']:<18} {valid:>10} "
            f"{format_number(row['avg_description_length']):>8} "
            f"{format_number(row['avg_completion_tokens']):>8}"
        )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Сетка параметров на 20 карточках dev.jsonl",
    )
    parser.add_argument(
        "--backend",
        choices=("ollama", "openrouter"),
        default="openrouter",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Имя модели. Для ollama по умолчанию qwen2.5:1.5b",
    )
    parser.add_argument("--input", default="dev.jsonl")
    parser.add_argument("--limit", type=int, default=CARD_LIMIT)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()
    if not args.model:
        if args.backend == "ollama":
            args.model = "qwen2.5:1.5b"
        else:
            args.model = os.getenv("OPENROUTER_MODEL")
    if not args.model:
        parser.error("укажите --model или переменную OPENROUTER_MODEL")
    return args


def main() -> None:
    args = parse_args()
    cards = load_cards(args.input)[: args.limit]
    if len(cards) < args.limit:
        raise RuntimeError(
            f"В {args.input} карточек {len(cards)}, нужно хотя бы {args.limit}"
        )

    runs = build_runs()
    logger.info(
        "Модель: %s, карточек: %d, комбинаций: %d",
        args.model,
        len(cards),
        len(runs),
    )

    table: list[dict] = []
    stored_runs = []
    for run in runs:
        outcomes = [
            evaluate_card(card, args.model, run["config"], args.backend)
            for card in cards
        ]
        valid_count = sum(1 for item in outcomes if item["valid"])
        logger.info(
            "config=%s valid=%d/%d",
            run["config"],
            valid_count,
            len(cards),
        )
        stored_runs.append(
            {
                "config": run["config"],
                "outcomes": outcomes,
            }
        )
        for row in run["rows"]:
            table.append(summarize(row["parameter"], row["value"], outcomes))

    by_parameter = {
        parameter: [row for row in table if row["parameter"] == parameter]
        for parameter in SWEEP
    }
    chosen_rows = {
        "temperature": choose_by_rate(by_parameter["temperature"], BASELINE["temperature"]),
        "top_p": choose_by_rate(by_parameter["top_p"], BASELINE["top_p"]),
        "top_k": choose_by_rate(by_parameter["top_k"], BASELINE["top_k"]),
        "repetition_penalty": choose_penalty(by_parameter["repetition_penalty"]),
        "max_tokens": choose_max_tokens(by_parameter["max_tokens"]),
    }
    chosen = {parameter: row["value"] for parameter, row in chosen_rows.items()}
    justification = justify(table, chosen_rows)

    report = {
        "backend": args.backend,
        "model": args.model,
        "product_ids": [card["product_id"] for card in cards],
        "baseline": BASELINE,
        "table": table,
        "chosen": chosen,
        "justification": justification,
        "runs": stored_runs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_table(table)
    print()
    print("Выбранный набор:")
    for parameter, value in chosen.items():
        print(f"  {parameter}: {format_value(parameter, value)}")
    print()
    for line in justification:
        print(line)
    logger.info("Отчёт записан в %s", output_path)


if __name__ == "__main__":
    main()
