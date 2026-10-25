import json
from pathlib import Path


def load_cards(path: str = "dev.jsonl") -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def format_card(index: int | None, card: dict) -> str:
    attributes = ", ".join(f"{key}: {value}" for key, value in card["attributes"].items())
    reviews = card["reviews"]
    header = f"{card['product_id']} | {card['title']}"
    if index is not None:
        header = f"{index:02}. {header}"
    lines = [
        header,
        f"    категория: {card['category']}",
        f"    атрибуты: {attributes}",
        f"    отзывов: {len(reviews)}",
    ]
    for review in reviews:
        lines.append(f"    - {review}")
    return "\n".join(lines)


def print_card(card: dict | str, path: str = "dev.jsonl") -> None:
    """Выводит одну карточку: словарь или product_id."""
    if isinstance(card, str):
        product_id = card
        for index, item in enumerate(load_cards(path), start=1):
            if item["product_id"] == product_id:
                print(format_card(index, item))
                return
        raise KeyError(f"Карточка {product_id} не найдена")
    print(format_card(None, card))


def main() -> None:
    cards = load_cards()
    print(f"Карточек: {len(cards)}\n")
    for index, card in enumerate(cards, start=1):
        print(format_card(index, card))
        print()


if __name__ == "__main__":
    main()
