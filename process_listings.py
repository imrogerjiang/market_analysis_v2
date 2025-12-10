#!/usr/bin/env python3
"""Post-process cleaned listings with generation metadata and run stats."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

LookupKey = Tuple[str, str]
GenRange = Tuple[int, int, str]


DEFAULT_GEN_LOOKUP_PATH = Path("lookups/gen_lookups.csv")


def parse_year(value: str) -> Optional[int]:
    value = value.strip()
    if value.isdigit():
        return int(value)
    return None


def load_gen_lookup(path: Path) -> Dict[LookupKey, List[GenRange]]:
    """Load generation ranges indexed by (make, model) in lower case."""
    lookup: DefaultDict[LookupKey, List[GenRange]] = defaultdict(list)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("gen_lookups.csv must have a header row")

        for row in reader:
            make = (row.get("make") or "").strip().lower()
            model = (row.get("model") or "").strip().lower()
            gen = (row.get("gen") or "").strip()
            year_start = parse_year(row.get("year_start") or "")
            year_end = parse_year(row.get("year_end") or "")

            if not (make and model and gen):
                continue

            if year_start is None or year_end is None:
                continue

            lookup[(make, model)].append((year_start, year_end, gen))

    # Sort to provide predictable matching when no range aligns (use earliest first)
    for key, ranges in lookup.items():
        lookup[key] = sorted(ranges, key=lambda r: (r[0], r[1]))

    return dict(lookup)


def model_gen_value(model: str, gen: str) -> str:
    if model and gen:
        return f"{model}_{gen}"
    if model:
        return model
    if gen:
        return gen
    return ""


def match_generation(
    gen_lookup: Dict[LookupKey, List[GenRange]], make: str, model: str, year: Optional[str]
) -> str:
    key = (make.strip().lower(), model.strip().lower())
    ranges = gen_lookup.get(key, [])
    if not ranges:
        return ""

    parsed_year = parse_year(year or "") if year else None
    if parsed_year is not None:
        for start, end, gen in ranges:
            if start <= parsed_year <= end:
                return gen

    # If no year match, fall back to the most recent generation
    return ranges[-1][2]


def enrich_rows(
    rows: Iterable[dict],
    gen_lookup: Dict[LookupKey, List[GenRange]],
    marketplace: str,
    scrape_date: str,
) -> List[dict]:
    """Attach generation, model_gen, date_scraped, and marketplace columns."""
    enriched: List[dict] = []
    for row in rows:
        make_raw = row.get("make") or ""
        model_raw = row.get("model") or ""
        year_raw = row.get("year") or ""

        gen = match_generation(gen_lookup, make_raw, model_raw, year_raw)

        row = dict(row)  # copy to avoid mutating input
        row["gen"] = gen
        row["model_gen"] = model_gen_value(model_raw, gen)
        row["date_scraped"] = scrape_date
        row["marketplace"] = marketplace
        enriched.append(row)

    return enriched


def write_output(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: List[dict]) -> None:
    total = len(rows)
    missing_gen = sum(1 for row in rows if not (row.get("gen") or "").strip())
    missing_make_model = sum(1 for row in rows if not ((row.get("make") or "").strip() and (row.get("model") or "").strip()))
    print(f"Total rows processed: {total}")
    print(f"Rows missing generation match: {missing_gen}")
    print(f"Rows missing make and/or model: {missing_make_model}")


def process_listings(
    input_path: Path, gen_lookup_path: Path, marketplace: str, output_path: Path, scrape_date: str
) -> None:
    marketplace_key = marketplace.strip().lower()
    if marketplace_key not in {"fb", "cs"}:
        raise ValueError("marketplace must be 'fb' or 'cs'")

    gen_lookup = load_gen_lookup(gen_lookup_path)

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV must include a header row")

        base_fields = list(reader.fieldnames)
        extra_fields = ["gen", "model_gen", "date_scraped", "marketplace"]
        fieldnames = base_fields + extra_fields

        enriched_rows = enrich_rows(reader, gen_lookup, marketplace_key, scrape_date)

    write_output(output_path, fieldnames, enriched_rows)
    summarize_rows(enriched_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich cleaned listings with generation metadata")
    parser.add_argument("input_csv", type=Path, help="Cleaned listings CSV to enrich")
    parser.add_argument(
        "gen_lookup",
        type=Path,
        nargs="?",
        default=DEFAULT_GEN_LOOKUP_PATH,
        help=f"CSV containing make/model to gen mapping (default: {DEFAULT_GEN_LOOKUP_PATH})",
    )
    parser.add_argument("marketplace", choices=["fb", "cs"], help="Marketplace code (fb or cs)")
    parser.add_argument(
        "output_csv",
        type=Path,
        nargs="?",
        help="Optional output path (defaults to <input>_processed.csv)",
    )
    parser.add_argument(
        "--date",
        dest="scrape_date",
        type=str,
        default=_dt.date.today().isoformat(),
        help="Override the date_scraped value (default: today in ISO format)",
    )
    return parser


def main(argv: List[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = args.output_csv or args.input_csv.with_name(f"{args.input_csv.stem}_processed.csv")

    process_listings(args.input_csv, args.gen_lookup, args.marketplace, output_path, args.scrape_date)
    print(f"Wrote enriched data to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
