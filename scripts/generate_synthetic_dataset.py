#!/usr/bin/env python3
"""Generate the synthetic interview dataset with one GPT call per answer."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "synthetic_interviews.jsonl"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 1.0

COMPANY_NAME = "Hungarian Coal Mining Industrial Association"

QUESTIONS = [
    "Overall, how would you describe your experience working here?",
    "How effective is communication between leadership and employees?",
    "How fair do compensation, benefits, and recognition feel?",
    "What opportunities do you see for career development or skill growth?",
    "How would you describe collaboration and trust within your immediate team?",
    "How manageable are workload, scheduling, and work-life balance?",
    "What is your biggest day-to-day frustration, and what would you change?",
    "How supportive is your direct manager?",
    "Which company values or cultural norms are visible in daily work?",
    "What one improvement would most help the organization?",
]

SITES = {
    "Site A": {
        "strength": "stable routines and a strong safety mindset",
        "friction": "slow approvals between departments",
        "opportunity": "clearer ownership for cross-site decisions",
    },
    "Site B": {
        "strength": "energetic collaboration and practical problem solving",
        "friction": "priorities that change late in the week",
        "opportunity": "more consistent planning discipline",
    },
    "Site C": {
        "strength": "close local support and dependable operational knowledge",
        "friction": "shift planning pressure during maintenance periods",
        "opportunity": "better coordination between office planning and field reality",
    },
}

MOODS = [
    "happy",
    "hopeful",
    "calm",
    "focused",
    "inspired",
    "curious",
    "proud",
    "thoughtful",
    "motivated",
    "grateful",
    "patient",
    "optimistic",
    "reflective",
    "a little tired",
    "a bit sad",
    "mildly concerned",
    "quietly confident",
]

ANSWER_ANGLES = [
    "daily routines",
    "communication timing",
    "recognition",
    "career growth",
    "team trust",
    "workload pressure",
    "decision follow-through",
    "manager support",
    "safety culture",
    "practical improvement",
]

PERSONA_BY_INTERVIEWEE = {
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

PERSONA_PROFILES = {
    ("white", "manager"): {
        "lens": "I see the workplace through planning, stakeholder alignment, and risk control",
        "positive": "strategic communication is improving",
        "concern": "decision cycles can become too cautious",
        "need": "faster prioritization and cleaner escalation paths",
        "communication": (
            "As a manager, I usually hear the business reason earlier than my team, but the cascade still needs "
            "clearer owners and timing."
        ),
    },
    ("white", "employee"): {
        "lens": "I experience the company through project work, peer collaboration, and development chances",
        "positive": "team cooperation is usually constructive",
        "concern": "career development can feel informal and uneven",
        "need": "clearer growth paths and more specific feedback",
        "communication": (
            "As an employee, I often hear the decision after the tradeoffs are settled, so the missing part is "
            "early context and space to ask questions."
        ),
    },
    ("blue", "manager"): {
        "lens": "I focus on crews, safe execution, and keeping daily operations predictable",
        "positive": "local teams have useful autonomy",
        "concern": "administration takes time away from coaching and field support",
        "need": "simpler reporting and earlier input into plans",
        "communication": (
            "As a manager close to the field, I need messages early enough to adjust crew plans instead of simply "
            "explaining a finished decision."
        ),
    },
    ("blue", "employee"): {
        "lens": "I judge the workplace by direct teamwork, fair schedules, and practical support",
        "positive": "people on the crew help each other without making a big show of it",
        "concern": "pay progression and shift changes can feel hard to influence",
        "need": "more predictable schedules and visible recognition for hands-on work",
        "communication": (
            "As an employee on the crew, I care less about polished announcements and more about what changes on "
            "the next shift."
        ),
    },
}

INTERVIEWEE_TRAITS = {
    1: {
        "focus": "handoffs from planning into execution",
        "frustration": "waiting for signatures after everyone already agrees",
        "recognition": "people who unblock others before a deadline slips",
        "growth": "cross-site leadership exposure",
        "balance": "meeting-heavy days that leave analysis until late afternoon",
        "voice": "I tend to look for the owner, the deadline, and the risk before I judge a change.",
    },
    2: {
        "focus": "budget choices and resource tradeoffs",
        "frustration": "decisions reopening after teams have already prepared",
        "recognition": "quiet coordination work that prevents escalation",
        "growth": "more practice leading complex stakeholder reviews",
        "balance": "urgent approvals that crowd out coaching time",
        "voice": "My first question is usually whether the right people were involved early enough.",
    },
    3: {
        "focus": "project coordination with several departments",
        "frustration": "late changes that make earlier work feel wasted",
        "recognition": "specific feedback after a difficult delivery",
        "growth": "a clearer path from specialist work into senior responsibility",
        "balance": "context switching between planned tasks and sudden requests",
        "voice": "I notice whether communication helps me prioritize or just adds another message to track.",
    },
    4: {
        "focus": "daily collaboration with peers and internal customers",
        "frustration": "unclear ownership when a request crosses departments",
        "recognition": "managers naming the actual contribution, not just saying thanks",
        "growth": "regular mentoring and better visibility into open roles",
        "balance": "last-minute requests that arrive without the background story",
        "voice": "I value direct explanations because they help me decide what to do first.",
    },
    5: {
        "focus": "data quality, reporting, and follow-through",
        "frustration": "reporting the same issue in several formats",
        "recognition": "being trusted to fix a process instead of only reporting on it",
        "growth": "training that connects tools with real business decisions",
        "balance": "peaks around reporting deadlines",
        "voice": "I listen for whether a message includes the reason, the metric, and the next check-in.",
    },
    6: {
        "focus": "supporting colleagues while protecting my own workload",
        "frustration": "feedback arriving only when something is already delayed",
        "recognition": "clear appreciation for reliable everyday work",
        "growth": "more frequent one-to-one feedback on skill progress",
        "balance": "helping others while my own priorities keep moving",
        "voice": "Communication lands best for me when it is plain, timely, and specific.",
    },
    7: {
        "focus": "safe crew execution during changing operational plans",
        "frustration": "paperwork pulling attention away from people in the field",
        "recognition": "foremen and crew leads who prevent safety shortcuts",
        "growth": "coaching support for newer supervisors",
        "balance": "being available for the crew after long operational days",
        "voice": "I judge a message by whether I can turn it into a safe work sequence.",
    },
    8: {
        "focus": "maintenance windows, staffing, and equipment readiness",
        "frustration": "plans that look finished before field constraints are tested",
        "recognition": "early warnings from crews that save time later",
        "growth": "more say in planning before maintenance decisions are locked",
        "balance": "schedule changes that ripple into weekends",
        "voice": "The best updates help me protect both output and safety at the same time.",
    },
    9: {
        "focus": "shift routines and practical help between crew members",
        "frustration": "schedule changes that are explained after the fact",
        "recognition": "visible credit for hands-on problem solving",
        "growth": "fair access to training on newer equipment",
        "balance": "maintenance periods that make family planning harder",
        "voice": "I want to know what changes for my shift, who decided it, and who can answer questions.",
    },
    10: {
        "focus": "tool availability, safe pace, and clear instructions",
        "frustration": "being asked for input after the plan is already set",
        "recognition": "small but concrete thanks when someone covers a difficult job",
        "growth": "a visible route from reliable crew work into a lead role",
        "balance": "overtime that is manageable only when we know early",
        "voice": "A short briefing works well when it respects what the crew already knows.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--model", default=os.getenv("OPENAI_CHAT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for repeatable mood selection.")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep after each API call.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep rows already present in the output file and generate only missing ids.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate deterministic placeholder answers without calling the OpenAI API.",
    )
    return parser.parse_args()


def persona_label(collar: str, role: str) -> str:
    collar_label = "white-collar" if collar == "white" else "blue-collar"
    return f"{collar_label} {role}"


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return ""

    try:
        import tomllib
    except ModuleNotFoundError:
        for line in secrets_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "OPENAI_API_KEY":
                return value.strip().strip("\"'")
        return ""

    with secrets_path.open("rb") as handle:
        secrets = tomllib.load(handle)
    return secrets.get("OPENAI_API_KEY", "")


def build_prompt(site: str, interviewee_no: int, question_no: int, mood: str) -> str:
    collar, role = PERSONA_BY_INTERVIEWEE[interviewee_no]
    profile = PERSONA_PROFILES[(collar, role)]
    site_profile = SITES[site]
    trait = INTERVIEWEE_TRAITS[interviewee_no]
    global_interviewee_no = list(SITES).index(site) * 10 + interviewee_no
    angle = ANSWER_ANGLES[(global_interviewee_no + question_no) % len(ANSWER_ANGLES)]
    question = QUESTIONS[question_no - 1]

    return (
        f"Give a one sentence answer to the following question as a person working for "
        f"{COMPANY_NAME}, your role is {role} and {collar}-collar, your mood is {mood}. "
        f"The question is the following: {question}\n\n"
        f"Keep this answer clearly different from the other synthetic answers. You are fictional interviewee "
        f"{global_interviewee_no} of 30, also known as {site} interviewee #{interviewee_no}. Your site strength is "
        f"{site_profile['strength']}; your site friction is {site_profile['friction']}; your site improvement "
        f"opportunity is {site_profile['opportunity']}. Your personal focus is {trait['focus']}; your usual "
        f"frustration is {trait['frustration']}; your recognition signal is {trait['recognition']}; your growth "
        f"interest is {trait['growth']}; your workload pressure is {trait['balance']}. Your broader role lens is: "
        f"{profile['lens']}. For this answer, emphasize {angle} without naming the mood."
    )


def system_prompt() -> str:
    return (
        "You generate fictional employee interview answers for a public demo dataset. Return exactly one sentence, "
        "plain text only, with no quotation marks, bullets, labels, or preamble. Vary sentence structure, vocabulary, "
        "and concrete details from answer to answer. Do not mention that the data is synthetic. Do not mention real "
        "Hungarian companies, towns, public figures, or identifiable people."
    )


def normalize_answer(answer: str) -> str:
    answer = " ".join(answer.strip().strip("\"'").split())
    return answer


def generate_answer(
    client: OpenAI,
    *,
    model: str,
    temperature: float,
    site: str,
    interviewee_no: int,
    question_no: int,
    mood: str,
    max_retries: int,
) -> str:
    prompt = build_prompt(site, interviewee_no, question_no, mood)
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                presence_penalty=0.7,
                frequency_penalty=0.4,
                max_tokens=90,
            )
            answer = response.choices[0].message.content or ""
            return normalize_answer(answer)
        except Exception as exc:  # pragma: no cover - exercised only against the API.
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"OpenAI answer generation failed after {max_retries + 1} attempts: {last_error}") from last_error


def dry_run_answer(site: str, interviewee_no: int, question_no: int, _mood: str) -> str:
    collar, role = PERSONA_BY_INTERVIEWEE[interviewee_no]
    site_profile = SITES[site]
    trait = INTERVIEWEE_TRAITS[interviewee_no]
    persona = persona_label(collar, role)
    return (
        f"As a {persona} in {site}, I would connect question {question_no} to {trait['focus']}, because the issue of "
        f"{site_profile['friction']} still affects how clearly people can turn plans into practical work for "
        f"interviewee {interviewee_no}."
    )


def iter_row_specs(rng: random.Random):
    for site in SITES:
        for interviewee_no in range(1, 11):
            for question_no, question in enumerate(QUESTIONS, start=1):
                yield site, interviewee_no, question_no, question, rng.choice(MOODS)


def build_row(site: str, interviewee_no: int, question_no: int, question: str, answer: str) -> dict:
    site_slug = site.lower().replace(" ", "_")
    collar, role = PERSONA_BY_INTERVIEWEE[interviewee_no]
    persona = persona_label(collar, role)
    text = f"RESPONDENT: {site}, interviewee #{interviewee_no}, {persona}\nQUESTION: {question}\nANSWER: {answer}"
    return {
        "id": f"{site_slug}_i{interviewee_no:02d}_q{question_no:02d}",
        "site": site,
        "question_no": question_no,
        "interviewee_no": interviewee_no,
        "collar": collar,
        "role": role,
        "question": question,
        "answer": answer,
        "text": text,
    }


def load_existing_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    rows_by_id = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: invalid JSON: {exc}") from exc
            rows_by_id[row["id"]] = row
    return rows_by_id


def write_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = load_existing_rows(args.output) if args.resume else {}
    mode = "a" if args.resume and args.output.exists() else "w"
    api_key = get_openai_api_key()
    if not args.dry_run and not api_key:
        raise SystemExit("OPENAI_API_KEY is required unless --dry-run is used.")

    client = None if args.dry_run else OpenAI(api_key=api_key)
    generated_count = 0
    reused_count = 0

    with args.output.open(mode, encoding="utf-8") as handle:
        for site, interviewee_no, question_no, question, mood in iter_row_specs(rng):
            row_id = f"{site.lower().replace(' ', '_')}_i{interviewee_no:02d}_q{question_no:02d}"
            if row_id in existing_rows:
                reused_count += 1
                continue

            if args.dry_run:
                answer = dry_run_answer(site, interviewee_no, question_no, mood)
            else:
                assert client is not None
                answer = generate_answer(
                    client,
                    model=args.model,
                    temperature=args.temperature,
                    site=site,
                    interviewee_no=interviewee_no,
                    question_no=question_no,
                    mood=mood,
                    max_retries=args.max_retries,
                )
                if args.delay:
                    time.sleep(args.delay)

            row = build_row(site, interviewee_no, question_no, question, answer)
            write_row(handle, row)
            generated_count += 1
            print(f"Wrote {row_id} with mood '{mood}'", flush=True)

    total_count = reused_count + generated_count
    print(f"Wrote {total_count} synthetic interview chunks to {args.output}")
    if args.resume and reused_count:
        print(f"Reused {reused_count} existing chunks and generated {generated_count} new chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
