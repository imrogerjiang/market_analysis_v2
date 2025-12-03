#!/usr/bin/env python3
"""Normalize raw Facebook Marketplace CSV exports into a consistent schema.

This script inspects the cell contents to infer the correct columns since
header names often change between exports.

Usage:
    python clean_facebook.py data/facebook.csv [output_path]

The output CSV will contain the following columns:
    href, year, make, model, listed_price, odometer, location
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional


CURRENCY_RE = re.compile(r"\$|AUD|AU\$", re.IGNORECASE)
ODOMETER_RE = re.compile(r"km", re.IGNORECASE)
MARKETPLACE_URL_RE = re.compile(r"facebook\.com/marketplace/item")
YEAR_TITLE_RE = re.compile(r"^\s*(20\d{2}|19\d{2})\b")
STATE_CODES = {"VIC", "NSW", "QLD", "SA", "WA", "NT", "TAS", "ACT"}

Extractor = Callable[[str], bool]


def detect_column(rows: List[List[str]], predicate: Extractor) -> Optional[int]:
    if not rows:
        return None

    scores = []
    for idx in range(len(rows[0])):
        score = sum(1 for row in rows if idx < len(row) and predicate(row[idx]))
        scores.append((score, idx))

    best_score, best_idx = max(scores, key=lambda pair: pair[0])
    return best_idx if best_score > 0 else None


def clean_price(value: str) -> str:
    cleaned = re.sub(r"[^0-9]", "", value)
    return cleaned if cleaned else ""


def clean_odometer(value: str) -> str:
    cleaned = re.sub(r"[^0-9]", "", value)
    return cleaned if cleaned else ""


def parse_vehicle_title(value: str) -> tuple[str, str, str]:
    match = YEAR_TITLE_RE.search(value)
    if not match:
        return "", "", value.strip()

    year = match.group(1)
    remainder = value[match.end():].strip()
    parts = remainder.split()
    if not parts:
        return year, "", ""

    make = parts[0]
    model = " ".join(parts[1:]) if len(parts) > 1 else ""
    return year, make, model


def choose_location_column(rows: List[List[str]]) -> Optional[int]:
    def looks_like_location(val: str) -> bool:
        stripped = val.strip()
        if not stripped:
            return False

        lowered = stripped.lower()
        if "http" in lowered or CURRENCY_RE.search(stripped) or ODOMETER_RE.search(stripped):
            return False
        if len(stripped) > 60:
            return False

        if stripped.upper() in STATE_CODES:
            return True
        if re.search(r"\b(?:NSW|VIC|QLD|SA|WA|NT|ACT|TAS)\b", stripped, re.IGNORECASE):
            return True

        return bool(re.search(r"[A-Za-z]+\s*,\s*[A-Za-z]+", stripped))

    return detect_column(rows, looks_like_location)


def load_rows(path: Path) -> List[List[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # discard header
        return list(reader)


def write_output(path: Path, rows: Iterable[List[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "href",
            "year",
            "make",
            "model",
            "listed_price",
            "odometer",
            "location",
        ])
        writer.writerows(rows)


def clean_facebook(input_path: Path, output_path: Path) -> None:
    rows = load_rows(input_path)
    if not rows:
        write_output(output_path, [])
        return

    href_idx = detect_column(rows, lambda v: bool(MARKETPLACE_URL_RE.search(v)))
    title_idx = detect_column(rows, lambda v: bool(YEAR_TITLE_RE.search(v)))
    price_idx = detect_column(rows, lambda v: bool(CURRENCY_RE.search(v)))
    odometer_idx = detect_column(rows, lambda v: bool(ODOMETER_RE.search(v)))
    location_idx = choose_location_column(rows)

    cleaned_rows: List[List[str]] = []
    for row in rows:
        href = row[href_idx] if href_idx is not None and href_idx < len(row) else ""

        title_val = row[title_idx] if title_idx is not None and title_idx < len(row) else ""
        year, make, model = parse_vehicle_title(title_val)

        price_raw = row[price_idx] if price_idx is not None and price_idx < len(row) else ""
        listed_price = clean_price(price_raw)

        odometer_raw = row[odometer_idx] if odometer_idx is not None and odometer_idx < len(row) else ""
        odometer = clean_odometer(odometer_raw)

        location = row[location_idx] if location_idx is not None and location_idx < len(row) else ""

        cleaned_rows.append([
            href.strip(),
            year,
            make,
            model,
            listed_price,
            odometer,
            location.strip(),
        ])

    columns = [
        "href",
        "year",
        "make",
        "model",
        "listed_price",
        "odometer",
        "location",
    ]

    print(f"{input_path.name} - Total rows: {len(cleaned_rows)}")
    for idx, column in enumerate(columns):
        nulls = sum(1 for row in cleaned_rows if idx >= len(row) or not str(row[idx]).strip())
        print(f"{column}: {nulls}")

    write_output(output_path, cleaned_rows)


def main(argv: List[str]) -> int:
    if not argv:
        print("Usage: python clean_facebook.py <input_csv> [output_csv]", file=sys.stderr)
        return 1

    input_path = Path(argv[0])
    if len(argv) > 1:
        output_path = Path(argv[1])
    else:
        output_path = input_path.with_name(f"{input_path.stem}_cleaned.csv")

    clean_facebook(input_path, output_path)
    print(f"Wrote cleaned data to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
