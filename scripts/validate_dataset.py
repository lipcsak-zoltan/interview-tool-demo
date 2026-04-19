#!/usr/bin/env python3
"""Validate the public demo JSONL dataset before rebuilding Chroma."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT / "data" / "synthetic_interviews.jsonl"

EXPECTED_FIELDS = [
    "id",
    "site",
    "question_no",
    "interviewee_no",
    "collar",
    "role",
    "question",
    "answer",
    "text",
]
EXPECTED_SITES = {"Site A", "Site B", "Site C"}
EXPECTED_QUESTIONS = set(range(1, 11))
EXPECTED_COLLARS = {"white", "blue"}
EXPECTED_ROLES = {"manager", "employee"}
EXPECTED_PERSONA_BY_INTERVIEWEE = {
    1: ("white", "manager"),
    2: ("white", "manager"),
    3: ("white", "employee"),
    4: ("white", "employee"),
    5: ("white", "employee"),
    6: ("white", "employee"),
    7: ("blue", "manager"),
    8: ("blue", "manager"),
    9: ("blue", "employee"),
    10: ("blue", "employee"),
}
EXPECTED_PERSONA_COUNTS_PER_SITE = {
    ("white", "manager"): 2,
    ("white", "employee"): 4,
    ("blue", "manager"): 2,
    ("blue", "employee"): 2,
}
BANNED_TERMS = [
    "FG" + "Sz",
    "F" + "\u00f6ldg\u00e1zsz\u00e1ll\u00edt\u00f3",
    "M" + "OL",
    "Si" + "\u00f3fok",
    "Kecske" + "m" + "\u00e9t",
    "Mis" + "kolc",
]


def load_rows(path: Path = DEFAULT_DATASET_PATH) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_no}: invalid JSON: {exc}") from exc
    return rows


def validate_rows(rows: list[dict]) -> list[str]:
    errors = []
    ids = [row.get("id") for row in rows]
    answers = [row.get("answer") for row in rows]

    if len(rows) != 300:
        errors.append(f"Expected 300 chunks, found {len(rows)}.")

    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        errors.append(f"Duplicate ids found: {duplicate_ids[:10]}")

    duplicate_answers = [item for item, count in Counter(answers).items() if count > 1]
    if duplicate_answers:
        errors.append(f"Duplicate answer texts found: {len(duplicate_answers)} repeated answer(s).")

    rows_by_site_interviewee = defaultdict(list)
    rows_by_site_question = defaultdict(list)
    unique_interviewees_by_site = defaultdict(dict)

    for index, row in enumerate(rows, start=1):
        if list(row.keys()) != EXPECTED_FIELDS:
            errors.append(f"Row {index} has unexpected fields: {list(row.keys())}")
            continue

        row_text = json.dumps(row, ensure_ascii=False).lower()
        for term in BANNED_TERMS:
            if term.lower() in row_text:
                errors.append(f"Row {index} contains a banned real-world term.")

        site = row["site"]
        question_no = row["question_no"]
        interviewee_no = row["interviewee_no"]
        collar = row["collar"]
        role = row["role"]

        if site not in EXPECTED_SITES:
            errors.append(f"Row {index} has invalid site: {site}")
        if question_no not in EXPECTED_QUESTIONS:
            errors.append(f"Row {index} has invalid question_no: {question_no}")
        if not isinstance(interviewee_no, int) or interviewee_no < 1 or interviewee_no > 10:
            errors.append(f"Row {index} has invalid interviewee_no: {interviewee_no}")
        if collar not in EXPECTED_COLLARS:
            errors.append(f"Row {index} has invalid collar: {collar}")
        if role not in EXPECTED_ROLES:
            errors.append(f"Row {index} has invalid role: {role}")

        expected_persona = EXPECTED_PERSONA_BY_INTERVIEWEE.get(interviewee_no)
        if expected_persona and (collar, role) != expected_persona:
            errors.append(
                f"Row {index} has persona {(collar, role)} for interviewee {interviewee_no}; "
                f"expected {expected_persona}."
            )

        collar_label = "white-collar" if collar == "white" else "blue-collar"
        persona = f"{collar_label} {role}"
        expected_text = (
            f"RESPONDENT: {site}, interviewee #{interviewee_no}, {persona}\n"
            f"QUESTION: {row['question']}\n"
            f"ANSWER: {row['answer']}"
        )
        if row["text"] != expected_text:
            errors.append(f"Row {index} text field does not match question/answer.")

        rows_by_site_interviewee[(site, interviewee_no)].append(row)
        rows_by_site_question[(site, question_no)].append(row)
        unique_interviewees_by_site[site][interviewee_no] = (collar, role)

    for site in EXPECTED_SITES:
        site_interviewees = unique_interviewees_by_site[site]
        if set(site_interviewees) != set(range(1, 11)):
            errors.append(f"{site} does not contain interviewees 1-10.")

        persona_counts = Counter(site_interviewees.values())
        if persona_counts != EXPECTED_PERSONA_COUNTS_PER_SITE:
            errors.append(f"{site} persona counts are {dict(persona_counts)}, expected {EXPECTED_PERSONA_COUNTS_PER_SITE}.")

        for interviewee_no in range(1, 11):
            rows_for_interviewee = rows_by_site_interviewee[(site, interviewee_no)]
            question_numbers = {row["question_no"] for row in rows_for_interviewee}
            if len(rows_for_interviewee) != 10 or question_numbers != EXPECTED_QUESTIONS:
                errors.append(f"{site} interviewee {interviewee_no} does not have exactly questions 1-10.")

        for question_no in EXPECTED_QUESTIONS:
            if len(rows_by_site_question[(site, question_no)]) != 10:
                errors.append(f"{site} question {question_no} does not have exactly 10 answers.")

    return errors


def main() -> int:
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET_PATH
    rows = load_rows(dataset_path)
    errors = validate_rows(rows)
    if errors:
        print("Dataset validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Dataset validation passed: {len(rows)} chunks in {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
