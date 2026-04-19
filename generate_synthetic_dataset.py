#!/usr/bin/env python3
"""Generate the reviewed synthetic interview dataset for the public demo."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "synthetic_interviews.jsonl"

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
    },
    ("white", "employee"): {
        "lens": "I experience the company through project work, peer collaboration, and development chances",
        "positive": "team cooperation is usually constructive",
        "concern": "career development can feel informal and uneven",
        "need": "clearer growth paths and more specific feedback",
    },
    ("blue", "manager"): {
        "lens": "I focus on crews, safe execution, and keeping daily operations predictable",
        "positive": "local teams have useful autonomy",
        "concern": "administration takes time away from coaching and field support",
        "need": "simpler reporting and earlier input into plans",
    },
    ("blue", "employee"): {
        "lens": "I judge the workplace by direct teamwork, fair schedules, and practical support",
        "positive": "people on the crew help each other without making a big show of it",
        "concern": "pay progression and shift changes can feel hard to influence",
        "need": "more predictable schedules and visible recognition for hands-on work",
    },
}


def persona_label(collar: str, role: str) -> str:
    collar_label = "white-collar" if collar == "white" else "blue-collar"
    return f"{collar_label} {role}"


def answer_for(site: str, interviewee_no: int, question_no: int) -> str:
    collar, role = PERSONA_BY_INTERVIEWEE[interviewee_no]
    profile = PERSONA_PROFILES[(collar, role)]
    site_profile = SITES[site]
    persona = persona_label(collar, role)
    variant = "first" if interviewee_no % 2 else "second"

    if question_no == 1:
        return (
            f"As a {persona}, my overall experience is mixed but credible. {site} benefits from "
            f"{site_profile['strength']}, and that gives daily work a dependable base. At the same time, "
            f"{profile['concern']} and {site_profile['friction']} can make progress feel heavier than it needs to be. "
            f"The {variant} thing I notice is that motivation stays high when people can see how their work matters."
        )
    if question_no == 2:
        return (
            f"Leadership communication is useful when it explains the reason behind a decision, not only the deadline. "
            f"In {site}, the strongest messages are about {site_profile['strength']}. The weaker part is that updates "
            f"arrive after practical choices have already been made. For my role, {profile['need']} would make "
            f"communication feel more like a working tool."
        )
    if question_no == 3:
        return (
            f"Compensation feels acceptable but not always connected to effort in a visible way. Recognition is strongest "
            f"inside the immediate team, where people notice who solved a hard problem or covered a difficult shift. "
            f"The concern is that {profile['concern']}, so appreciation can feel private rather than organizational. "
            f"I would not call it unfair, but I would call it uneven."
        )
    if question_no == 4:
        return (
            f"Growth opportunities exist, but they depend too much on the manager and the current workload. "
            f"{profile['lens'].capitalize()}, so I notice when learning is planned and when it is just improvised. "
            f"At {site}, people can learn a lot from experienced colleagues, but the path is not always written down. "
            f"A stronger development plan would help people prepare before a role becomes urgent."
        )
    if question_no == 5:
        return (
            f"Collaboration inside my immediate team is one of the better parts of the job. People share information, "
            f"cover gaps, and usually assume good intent. Trust becomes weaker across departments when "
            f"{site_profile['friction']} shows up. The team culture is healthy, but cross-functional work needs "
            f"more shared context and fewer last-minute corrections."
        )
    if question_no == 6:
        return (
            f"Workload is manageable in normal weeks and difficult when several priorities land at once. The pattern at "
            f"{site} is shaped by {site_profile['friction']}. {profile['positive'].capitalize()}, which helps people "
            f"absorb pressure for a while. The risk is that goodwill becomes the backup plan, especially when schedules "
            f"or approvals change late."
        )
    if question_no == 7:
        return (
            f"My biggest frustration is avoidable rework. People usually want to do the right thing, but plans sometimes "
            f"move before the people affected by them have enough input. I would change the handoff process so decisions "
            f"include operational constraints earlier. That would support {site_profile['opportunity']} and reduce the "
            f"feeling that problems are solved twice."
        )
    if question_no == 8:
        return (
            f"My direct manager is generally supportive and available when priorities are clear. The best support is "
            f"practical: removing blockers, translating expectations, and backing the team when tradeoffs are real. "
            f"The support is less effective when {profile['concern']}. I value direct feedback more than general praise "
            f"because it helps me decide what to improve next."
        )
    if question_no == 9:
        return (
            f"The most visible values are reliability, safety, and helping colleagues finish the work properly. Those "
            f"values show up in small daily habits, not slogans. {profile['positive'].capitalize()}, but the culture "
            f"also accepts too many slow processes as normal. The company is at its best when practical knowledge and "
            f"office planning respect each other."
        )
    if question_no == 10:
        return (
            f"The single improvement I would choose is better follow-through after decisions are announced. People need "
            f"to know who owns the next step, what changed, and how success will be checked. This would address "
            f"{site_profile['friction']} without changing the whole organization again. It would also make "
            f"{profile['need']} easier to achieve."
        )
    raise ValueError(f"Unsupported question number: {question_no}")


def build_rows() -> list[dict]:
    rows = []
    for site in SITES:
        site_slug = site.lower().replace(" ", "_")
        for interviewee_no in range(1, 11):
            collar, role = PERSONA_BY_INTERVIEWEE[interviewee_no]
            for question_no, question in enumerate(QUESTIONS, start=1):
                answer = answer_for(site, interviewee_no, question_no)
                text = f"QUESTION: {question}\nANSWER: {answer}"
                rows.append(
                    {
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
                )
    return rows


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} synthetic interview chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
