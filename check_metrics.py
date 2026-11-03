"""Пересчёт valid_rate и errors_by_reason по outputs/predictions.jsonl.

Цифры времени и цены берутся из секции run в outputs/report.json:
throughput = n / elapsed_sec, time_for_300 = 300 / throughput,
cost = (elapsed_sec / 3600 * 65) / n * 1000.
Качество считается заново по строкам jsonl, без Colab.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

PREDICTIONS_PATH = Path("outputs/predictions.jsonl")
BENCHMARK_PATH = Path("benchmark.jsonl")
REPORT_PATH = Path("outputs/report.json")
GPU_RUB_PER_HOUR = 65
REQUIRED_FIELDS = ("product_id", "description", "pros", "cons", "tags")
LIST_LIMITS = {"pros": (2, 5), "cons": (1, 3), "tags": (3, 8)}


def count_sentences(text: str) -> int:
    """Число предложений. Точка внутри числа концом предложения не считается."""
    return len([part for part in re.split(r"[.!?]+(?!\d)", text) if part.strip()])


def validate_card(payload: object, product_id: str) -> list[str]:
    """Список нарушений. Пустой список — карточка валидна."""
    if not isinstance(payload, dict):
        return ["Ответ не является JSON-объектом"]

    errors = []
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        errors.append(f"Нет полей: {', '.join(missing)}")

    if "product_id" in payload and payload["product_id"] != product_id:
        errors.append(
            f"product_id {payload['product_id']!r} не совпадает с {product_id!r}"
        )

    if "description" in payload:
        description = payload["description"]
        if not isinstance(description, str):
            errors.append("description должен быть строкой")
        else:
            length = len(description)
            if not 120 <= length <= 700:
                errors.append(f"description: {length} символов, нужно 120–700")
            sentences = count_sentences(description)
            if not 2 <= sentences <= 5:
                errors.append(f"description: {sentences} предложений, нужно 2–5")

    for name, (low, high) in LIST_LIMITS.items():
        if name not in payload:
            continue
        value = payload[name]
        if (
            not isinstance(value, list)
            or not all(isinstance(item, str) and item.strip() for item in value)
        ):
            errors.append(f"{name}: нужен список непустых строк")
            continue
        if not low <= len(value) <= high:
            errors.append(f"{name}: {len(value)} пунктов, нужно {low}–{high}")

    return errors


def primary_reason(record: dict, product_id: str) -> str | None:
    """Одна причина брака на карточку, в тех же ключах, что report.json.

    Строка только с product_id — так ноутбук записывает оборванный JSON.
    """
    if not isinstance(record, dict) or "description" not in record:
        return "Не JSON"
    errors = validate_card(record, product_id)
    if not errors:
        return None
    for error in errors:
        if "предложений" in error:
            return "предложений"
    for error in errors:
        if "JSON" in error:
            return "Не JSON"
    return errors[0]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            rows.append({"_parse_error": f"{error.msg} (строка {line_number})"})
            continue
        if not isinstance(payload, dict):
            rows.append({"_parse_error": f"не объект (строка {line_number})"})
            continue
        rows.append(payload)
    return rows


def quality_from_records(records: list[dict], expected_ids: list[str] | None = None) -> dict:
    """valid, valid_rate и errors_by_reason по уже прочитанным карточкам."""
    reasons: dict[str, int] = {}
    valid = 0
    aligned = expected_ids is not None and len(expected_ids) == len(records)
    for index, record in enumerate(records):
        if record.get("_parse_error"):
            reasons["Не JSON"] = reasons.get("Не JSON", 0) + 1
            continue
        product_id = record.get("product_id")
        if aligned:
            product_id = expected_ids[index]
        elif not isinstance(product_id, str):
            product_id = ""
        reason = primary_reason(record, product_id)
        if reason is None:
            valid += 1
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
    total = len(records)
    return {
        "n": total,
        "valid": valid,
        "valid_rate": (valid / total) if total else 0.0,
        "errors_by_reason": reasons,
    }


def formulas_from_run(run: dict) -> dict:
    """Метрики, которые выводятся из фактов прогона, а не из текста карточек."""
    total = run["n"]
    elapsed = run["elapsed_sec"]
    throughput = total / elapsed
    return {
        "throughput": throughput,
        "time_for_300_sec": 300 / throughput,
        "cost_vllm_per_1000": (elapsed / 3600 * GPU_RUB_PER_HOUR) / total * 1000,
    }


def build_metrics(run: dict, quality: dict) -> dict:
    derived = formulas_from_run(run)
    return {
        "valid": quality["valid"],
        "valid_rate": quality["valid_rate"],
        "throughput": derived["throughput"],
        "time_for_300_sec": derived["time_for_300_sec"],
        "cost_vllm_per_1000": derived["cost_vllm_per_1000"],
        "errors_by_reason": quality["errors_by_reason"],
    }


def merge_benchmark(report_path: Path, run: dict, quality: dict) -> dict:
    """Записать run и metrics. api_comparison не трогать."""
    report = {}
    if report_path.exists():
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            report = loaded
    report["run"] = run
    report["metrics"] = build_metrics(run, quality)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def check_report(report: dict, quality: dict) -> list[str]:
    """Расхождения между jsonl, формулами и уже записанным report.json."""
    problems = []
    run = report.get("run") or {}
    metrics = report.get("metrics") or {}
    if "n" not in run or "elapsed_sec" not in run:
        return ["В report.json нет run.n или run.elapsed_sec"]

    derived = formulas_from_run(run)
    for key, value in derived.items():
        stored = metrics.get(key)
        if stored is None or not _close(float(stored), value):
            problems.append(f"metrics.{key}: в файле {stored}, по формуле из run {value}")

    expected_n = run["n"]
    if quality["n"] != expected_n:
        problems.append(
            f"predictions.jsonl: {quality['n']} строк, run.n = {expected_n}. "
            "Нужна одна карточка на каждую строку benchmark.jsonl"
        )
        return problems

    if metrics.get("valid") != quality["valid"]:
        problems.append(f"metrics.valid: в файле {metrics.get('valid')}, по jsonl {quality['valid']}")
    stored_rate = metrics.get("valid_rate")
    if stored_rate is None or not _close(float(stored_rate), quality["valid_rate"]):
        problems.append(
            f"metrics.valid_rate: в файле {stored_rate}, по jsonl {quality['valid_rate']}"
        )
    if metrics.get("errors_by_reason") != quality["errors_by_reason"]:
        problems.append(
            "metrics.errors_by_reason: "
            f"в файле {metrics.get('errors_by_reason')}, по jsonl {quality['errors_by_reason']}"
        )
    return problems


def print_quality(quality: dict) -> None:
    print(
        f"Карточек: {quality['n']}. "
        f"Валидных: {quality['valid']}. "
        f"valid_rate: {quality['valid_rate']:.4f}."
    )
    if quality["errors_by_reason"]:
        print("Брак:", ", ".join(
            f"{reason} {count}" for reason, count in quality["errors_by_reason"].items()
        ))
    else:
        print("Брак: нет")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    if not args.predictions.exists():
        print(f"Нет файла {args.predictions}", file=sys.stderr)
        return 1

    records = load_jsonl(args.predictions)
    expected_ids = None
    if args.benchmark.exists():
        source = load_jsonl(args.benchmark)
        expected_ids = [row.get("product_id", "") for row in source]
        if len(source) != len(records):
            print(
                f"{args.benchmark.name}: {len(source)} строк, "
                f"{args.predictions.name}: {len(records)}."
            )
            expected_ids = None
    quality = quality_from_records(records, expected_ids)
    print_quality(quality)

    if not args.report.exists():
        print(f"Нет {args.report}, сверка пропущена")
        return 0

    report = json.loads(args.report.read_text(encoding="utf-8"))
    problems = check_report(report, quality)
    if not problems:
        print(f"{args.report} совпадает с jsonl и с формулами из run.")
        return 0
    print("Не сходится:")
    for problem in problems:
        print(f"- {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
