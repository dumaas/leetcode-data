#!/usr/bin/env python3
"""
This script generates a CSV with all the leetcode problems currently known.
"""

import argparse
import asyncio
import logging
import re
from html import unescape
from pathlib import Path
from typing import Dict, List

import polars as pl  # type: ignore
from tqdm import tqdm  # type: ignore

import leetcode_anki.helpers.leetcode

OUTPUT_FILE = "leetcode"


logging.getLogger().setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the script
    """
    parser = argparse.ArgumentParser(description="Generate Anki cards for leetcode")
    parser.add_argument(
        "--start", type=int, help="Start generation from this problem", default=0
    )
    parser.add_argument(
        "--stop", type=int, help="Stop generation on this problem", default=2**64
    )
    parser.add_argument(
        "--page-size",
        type=int,
        help="Get at most this many problems (decrease if leetcode API times out)",
        default=500,
    )
    parser.add_argument(
        "--list-id",
        type=str,
        help="Get all questions from a specific list id (https://leetcode.com/list?selectedList=<list_id>",
        default="",
    )
    parser.add_argument(
        "--output-file", type=str, help="Output filename", default=OUTPUT_FILE
    )

    args = parser.parse_args()

    return args


_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(s: str) -> str:
    """Remove HTML tags and unescape entities."""
    return _TAG_RE.sub("", unescape(s))


def normalize_difficulty(raw: str) -> str:
    """
    Normalize LeetCode difficulty HTML to one of: 'easy', 'medium', 'hard'.

    Examples:
      "<font color='green'>Easy</font>"  -> "easy"
      "  <b>Medium</b> "                 -> "medium"
    """
    text = strip_html(raw).strip().lower()
    if text not in {"easy", "medium", "hard"}:
        raise Exception(f"Unexpected difficulty: {text}")
    return text


async def generate_csv_row(
    leetcode_data: leetcode_anki.helpers.leetcode.LeetcodeData,
    leetcode_task_handle: str,
) -> dict:
    """Custom implementation WITHOUT genanki usage"""
    # Fetch all fields
    (
        problem_id,
        title,
        title_slug,
        category,
        description,
        difficulty_raw,
        paid,
        likes,
        dislikes,
        submissions_total,
        submissions_accepted,
        freq_bar,
        tags_raw,
    ) = await asyncio.gather(
        leetcode_data.problem_id(leetcode_task_handle),
        leetcode_data.title(leetcode_task_handle),
        leetcode_data.title_slug(leetcode_task_handle),
        leetcode_data.category(leetcode_task_handle),
        leetcode_data.description(leetcode_task_handle),
        leetcode_data.difficulty(leetcode_task_handle),
        leetcode_data.paid(leetcode_task_handle),
        leetcode_data.likes(leetcode_task_handle),
        leetcode_data.dislikes(leetcode_task_handle),
        leetcode_data.submissions_total(leetcode_task_handle),
        leetcode_data.submissions_accepted(leetcode_task_handle),
        leetcode_data.freq_bar(leetcode_task_handle),
        leetcode_data.tags(leetcode_task_handle),
    )
    # Normalize difficulty (HTML -> [easy|medium|hard])
    difficulty = normalize_difficulty(difficulty_raw)
    acceptance_rate = submissions_accepted / submissions_total * 100 if submissions_total else 0.0

    # Normalize tags as a proper list
    if isinstance(tags_raw, (list, tuple)):
        tags = list(tags_raw)
    elif tags_raw is None:
        tags = []
    else:
        tags = [str(tags_raw)]

    # Exclude difficulty from tags
    tags = [t for t in tags if not str(t).startswith("difficulty-")]

    return {
        "problem_id": problem_id,
        "title": title,
        "title_slug": title_slug,
        "category": category,
        "description": description,
        "difficulty": difficulty,
        "paid": bool(paid),
        "likes": likes,
        "dislikes": dislikes,
        "submissions_total": submissions_total,
        "submissions_accepted": submissions_accepted,
        "acceptance_rate": acceptance_rate,
        "freq_bar": freq_bar,
        "tags": tags,
    }


def sanitize_filename(name: str) -> str:
    """Remove characters that are problematic in filenames (Windows especially)."""
    return re.sub(r'[\\/*?:"<>|]', "", name)


def generate_obsidian_files(df: pl.DataFrame, out_dir: Path) -> None:
    """Generate one Obsidian-ready Markdown file per problem"""
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in df.iter_rows(named=True):
        problem_id = row["problem_id"]
        title = row["title"]
        slug = row["title_slug"]
        description_html = row["description"]
        difficulty = row["difficulty"]
        category = row["category"]
        paid = row["paid"]
        likes = row["likes"]
        dislikes = row["dislikes"]
        submissions_total = row["submissions_total"]
        submissions_accepted = row["submissions_accepted"]
        acceptance_rate = row["acceptance_rate"]
        freq_bar = row["freq_bar"]
        tags = row["tags"]

        # --- YAML-friendly tags (metadata only, lowercase slugs) ---
        if tags:
            # Multi-line list
            tags_block = "\n".join(f"- {t}" for t in tags)
            yaml_tags = f"tags:\n{tags_block}"
        else:
            # Valid YAML empty list
            yaml_tags = "tags: []"

        # --- Wikilinks (conceptual Obsidian tags) ---
        # Remove difficulty-* tags
        conceptual_tags = [
            t for t in tags
            if not str(t).startswith("difficulty-")
        ]

        # Convert slug-like tags to Pretty Names:
        #   "hash-table" -> "Hash Table"
        #   "two-pointers" -> "Two Pointers"
        def prettify_tag(t: str) -> str:
            return " ".join(word.capitalize() for word in t.replace("_", "-").split("-"))

        tag_wikilinks = [f"[[{prettify_tag(t)}]]" for t in conceptual_tags]

        # Build the Markdown list
        wikilinks_block = "\n".join(f"- {w}" for w in tag_wikilinks)

        safe_title = sanitize_filename(title)
        filename = out_dir / f"{problem_id} - {safe_title}.md"

        md = f"""---
id: {problem_id}
title: "{title}"
slug: {slug}
url: "https://leetcode.com/problems/{slug}/"
difficulty: {difficulty}
category: {category}
{yaml_tags}
paid: {str(paid).lower()}
likes: {likes}
dislikes: {dislikes}
submissions_total: {submissions_total}
submissions_accepted: {submissions_accepted}
acceptance_rate: {acceptance_rate}
freq_bar: {freq_bar}
lists: []
---

# {problem_id}. {title}

## Description

{description_html}

---

## Tags
{wikilinks_block}

---

## Links
- LeetCode: https://leetcode.com/problems/{slug}/
"""

        filename.write_text(md, encoding="utf-8")


async def generate_csv(
    start: int, stop: int, page_size: int, list_id: str, output_file: str
) -> None:
    """Custom implementation for generating CSV"""
    base_name = Path(output_file).name  # e.g. "leetcode"

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{base_name}.parquet"
    csv_path = output_dir / f"{base_name}.csv"
    obsidian_dir = output_dir / f"{base_name}_obsidian"

    leetcode_data = leetcode_anki.helpers.leetcode.LeetcodeData(
        start, stop, page_size, list_id
    )

    logging.info("Fetching problem handles")
    task_handles = await leetcode_data.all_problems_handles()

    logging.info("Generating CSV rows")

    # Create tasks so we can drive them with tqdm using as_completed
    tasks = [
        asyncio.create_task(generate_csv_row(leetcode_data, handle))
        for handle in task_handles
    ]

    rows: List[Dict[str, object]] = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), unit="problem"):
        row = await coro
        rows.append(row)
    
    df = pl.DataFrame(rows)
    df = df.select(
        "problem_id",
        "title",
        "title_slug",
        "category",
        "description",
        "difficulty",
        "paid",
        "likes",
        "dislikes",
        "submissions_total",
        "submissions_accepted",
        "acceptance_rate",
        "freq_bar",
        "tags",
    ).sort("problem_id")

    # 1) Write canonical Parquet
    logging.info("Writing Parquet to %s", parquet_path)
    df.write_parquet(str(parquet_path))

    # 2) Write flattened CSV (tags joined by ",")
    df_csv = df.with_columns(
        pl.col("tags").list.join(",").alias("tags")
    )
    logging.info("Writing CSV to %s", csv_path)
    df_csv.write_csv(str(csv_path))

    # 3) Write Obsidian markdown files
    logging.info("Writing Obsidian Markdown files to %s", obsidian_dir)
    generate_obsidian_files(df, obsidian_dir)

async def main() -> None:
    args = parse_args()

    await generate_csv(
        start=args.start,
        stop=args.stop,
        page_size=args.page_size,
        list_id=args.list_id,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    asyncio.run(main())
