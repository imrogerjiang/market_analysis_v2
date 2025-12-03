#!/usr/bin/env python3
"""Normalize raw carsales CSV exports into a consistent schema.

This script infers the required columns by looking at the cell contents
rather than relying on header names (which vary between exports).

Usage:
    python clean_carsales.py data/carsales.csv [output_path]

The output CSV will contain the following columns:
    href, year, make, model, trim, odometer, seller_type,
    location, listed_price
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional


@dataclass
class ColumnGuess:
    index: int
    score: int


STATE_CODES = {
    "VIC",
    "NSW",
    "QLD",
    "SA",
    "WA",
    "NT",
    "TAS",
    "ACT",
}


CURRENCY_RE = re.compile(r"\$|AUD|AU\$", re.IGNORECASE)
ODOMETER_RE = re.compile(r"km", re.IGNORECASE)
DETAIL_URL_RE = re.compile(r"carsales\.com\.au/cars/details")
YEAR_TITLE_RE = re.compile(r"^\s*(20\d{2}|19\d{2})\b")


Extractor = Callable[[str], bool]


def detect_column(rows: List[List[str]], predicate: Extractor) -> Optional[int]:
    """Return the index of the column with the highest predicate matches."""
    if not rows:
        return None

    match_scores: List[ColumnGuess] = []
    for idx in range(len(rows[0])):
        score = sum(1 for row in rows if idx < len(row) and predicate(row[idx]))
        match_scores.append(ColumnGuess(idx, score))

    best = max(match_scores, key=lambda g: g.score)
    if best.score == 0:
        return None
    return best.index


def clean_price(value: str) -> str:
    cleaned = re.sub(r"[^0-9]", "", value)
    return cleaned if cleaned else ""


def clean_odometer(value: str) -> str:
    cleaned = re.sub(r"[^0-9]", "", value)
    return cleaned if cleaned else ""


def parse_title_for_vehicle(value: str) -> tuple[str, str, str]:
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
        if stripped.upper() in STATE_CODES:
            return True
        # Allow short uppercase suburb/state strings
        return bool(re.fullmatch(r"[A-Z]{2,5}", stripped))

    return detect_column(rows, looks_like_location)


def choose_trim_column(rows: List[List[str]], title_idx: Optional[int], price_idx: Optional[int], odometer_idx: Optional[int]) -> Optional[int]:
    skip = {idx for idx in (title_idx, price_idx, odometer_idx) if idx is not None}

    def looks_like_trim(val: str) -> bool:
        if not val or val.strip() == "Read more":
            return False
        if CURRENCY_RE.search(val) or ODOMETER_RE.search(val):
            return False
        if DETAIL_URL_RE.search(val):
            return False
        if re.fullmatch(r"\d+", val.strip()):
            return False
        # Trim strings often contain drivetrain or grade keywords
        return bool(re.search(r"Auto|Manual|GT|Touring|Pure|Evolve|Neo|SP|Sport|G\d{2}|Maxx|Ascent", val, re.IGNORECASE))

    best_idx: Optional[int] = None
    best_score = 0
    for idx in range(len(rows[0])):
        if idx in skip:
            continue
        score = sum(1 for row in rows if idx < len(row) and looks_like_trim(row[idx]))
        if score > best_score:
            best_idx, best_score = idx, score
    return best_idx


def choose_seller_type(rows: List[List[str]]) -> Optional[int]:
    keywords = ("dealer", "private", "used")

    def looks_like_seller(val: str) -> bool:
        lower = val.lower()
        return any(k in lower for k in keywords)

    return detect_column(rows, looks_like_seller)


def load_rows(path: Path) -> List[List[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # discard header if present
        return list(reader)


def write_output(path: Path, rows: Iterable[List[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "href",
            "year",
            "make",
            "model",
            "trim",
            "odometer",
            "seller_type",
            "location",
            "listed_price",
        ])
        writer.writerows(rows)


def clean_carsales(input_path: Path, output_path: Path) -> None:
    rows = load_rows(input_path)
    if not rows:
        write_output(output_path, [])
        return

    href_idx = detect_column(rows, lambda v: bool(DETAIL_URL_RE.search(v)))
    title_idx = detect_column(rows, lambda v: bool(YEAR_TITLE_RE.search(v)))
    price_idx = detect_column(rows, lambda v: bool(CURRENCY_RE.search(v)))
    odometer_idx = detect_column(rows, lambda v: bool(ODOMETER_RE.search(v)))
    seller_idx = choose_seller_type(rows)
    location_idx = choose_location_column(rows)
    trim_idx = choose_trim_column(rows, title_idx, price_idx, odometer_idx)

    cleaned_rows: List[List[str]] = []
    for row in rows:
        href = row[href_idx] if href_idx is not None and href_idx < len(row) else ""

        title_val = row[title_idx] if title_idx is not None and title_idx < len(row) else ""
        year, make, model = parse_title_for_vehicle(title_val)

        trim = row[trim_idx] if trim_idx is not None and trim_idx < len(row) else ""
        odometer_raw = row[odometer_idx] if odometer_idx is not None and odometer_idx < len(row) else ""
        odometer = clean_odometer(odometer_raw)

        seller_type = row[seller_idx] if seller_idx is not None and seller_idx < len(row) else ""
        location = row[location_idx] if location_idx is not None and location_idx < len(row) else ""

        price_raw = row[price_idx] if price_idx is not None and price_idx < len(row) else ""
        listed_price = clean_price(price_raw)

        cleaned_rows.append([
            href.strip(),
            year,
            make,
            model,
            trim.strip(),
            odometer,
            seller_type.strip(),
            location.strip(),
            listed_price,
        ])

    columns = [
        "href",
        "year",
        "make",
        "model",
        "trim",
        "odometer",
        "seller_type",
        "location",
        "listed_price",
    ]

    print(f"{input_path.name} - Total rows: {len(cleaned_rows)}")
    for idx, column in enumerate(columns):
        nulls = sum(1 for row in cleaned_rows if idx >= len(row) or not str(row[idx]).strip())
        print(f"{column}: {nulls}")

    write_output(output_path, cleaned_rows)


def main(argv: List[str]) -> int:
    if not argv:
        print("Usage: python clean_carsales.py <input_csv> [output_csv]", file=sys.stderr)
        return 1

    input_path = Path(argv[0])
    if len(argv) > 1:
        output_path = Path(argv[1])
    else:
        output_path = input_path.with_name(f"{input_path.stem}_cleaned.csv")

    clean_carsales(input_path, output_path)
    print(f"Wrote cleaned data to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
