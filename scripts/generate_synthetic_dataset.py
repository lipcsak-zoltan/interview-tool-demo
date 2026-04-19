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

QUESTION_INTROS = {
    1: [
        "The experience is generally solid, although not effortless.",
        "I would describe the workplace as dependable, with some stubborn sources of drag.",
        "Most days feel purposeful, but progress depends on how well different parts of the organization line up.",
    ],
    2: [
        "Communication works best when it explains both the decision and the tradeoff behind it.",
        "The useful messages are the ones that arrive before people have already adjusted their own plans.",
        "I do not need every detail, but I do need the reason, the owner, and the timing.",
    ],
    3: [
        "Pay and recognition feel acceptable overall, but the signal is uneven.",
        "The formal package is not the main complaint; visibility of effort is the weaker point.",
        "People usually know who contributes, but the organization does not always show that it knows.",
    ],
}


def cycle(values: list[str], site: str, interviewee_no: int, question_no: int) -> str:
    site_offset = list(SITES).index(site)
    return values[(site_offset + interviewee_no + question_no) % len(values)]


def sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def persona_label(collar: str, role: str) -> str:
    collar_label = "white-collar" if collar == "white" else "blue-collar"
    return f"{collar_label} {role}"


def answer_for(site: str, interviewee_no: int, question_no: int) -> str:
    collar, role = PERSONA_BY_INTERVIEWEE[interviewee_no]
    profile = PERSONA_PROFILES[(collar, role)]
    site_profile = SITES[site]
    trait = INTERVIEWEE_TRAITS[interviewee_no]
    persona = persona_label(collar, role)

    if question_no == 1:
        intro = cycle(QUESTION_INTROS[1], site, interviewee_no, question_no)
        return (
            f"As a {persona}, {intro} {site} benefits from {site_profile['strength']}, which gives daily work "
            f"a dependable base. My own focus is {trait['focus']}, so I feel the friction when "
            f"{site_profile['friction']} combines with the fact that {profile['concern']}. {trait['voice']}"
        )
    if question_no == 2:
        intro = cycle(QUESTION_INTROS[2], site, interviewee_no, question_no)
        return (
            f"{intro} {profile['communication']} In {site}, the strongest messages are about "
            f"{site_profile['strength']}. The weak spot for me is {trait['frustration']}. Better communication would "
            f"mean {profile['need']} and a clearer link to {trait['focus']}."
        )
    if question_no == 3:
        intro = cycle(QUESTION_INTROS[3], site, interviewee_no, question_no)
        return (
            f"{intro} Recognition is strongest inside the immediate team, especially for {trait['recognition']}. "
            f"The concern is that {profile['concern']}, so appreciation can feel private rather than organizational. "
            f"I would not call it unfair, but I would like rewards and feedback to connect more clearly to "
            f"{trait['focus']}."
        )
    if question_no == 4:
        return (
            f"Growth opportunities exist, but they depend too much on the manager and the current workload. "
            f"{sentence_case(profile['lens'])}, so I notice when learning is planned and when it is improvised. "
            f"At {site}, people can learn a lot from experienced colleagues, but I would benefit most from "
            f"{trait['growth']}. A stronger development plan would help people prepare before a role becomes urgent."
        )
    if question_no == 5:
        return (
            f"Collaboration inside my immediate team is one of the better parts of the job. People share information, "
            f"cover gaps, and usually assume good intent around {trait['focus']}. Trust becomes weaker across "
            f"departments when {site_profile['friction']} shows up. The team culture is healthy, but cross-functional "
            f"work needs more shared context and fewer last-minute corrections."
        )
    if question_no == 6:
        return (
            f"Workload is manageable in normal weeks and difficult when several priorities land at once. The pattern at "
            f"{site} is shaped by {site_profile['friction']}. {sentence_case(profile['positive'])}, which helps people "
            f"absorb pressure for a while. My pressure point is {trait['balance']}. The risk is that goodwill becomes "
            f"the backup plan, especially when schedules or approvals change late."
        )
    if question_no == 7:
        return (
            f"My biggest frustration is {trait['frustration']}. People usually want to do the right thing, but plans "
            f"sometimes move before the people affected by them have enough input. I would change the handoff process "
            f"so decisions include operational constraints earlier. That would support {site_profile['opportunity']} "
            f"and reduce the feeling that problems are solved twice."
        )
    if question_no == 8:
        return (
            f"My direct manager is generally supportive and available when priorities are clear. The best support is "
            f"practical: removing blockers, translating expectations, and backing the team when tradeoffs are real. "
            f"At {site}, that support matters most when {site_profile['strength']} has to be protected despite "
            f"{site_profile['friction']}. The support is less effective when {profile['concern']}. For my work on "
            f"{trait['focus']}, I value direct feedback more than general praise because it helps me decide what to "
            f"improve next."
        )
    if question_no == 9:
        return (
            f"The most visible values are reliability, safety, and helping colleagues finish the work properly. Those "
            f"values show up in small daily habits, not slogans. At {site}, {site_profile['strength']} reinforces "
            f"those values. {sentence_case(profile['positive'])}, but the culture also accepts too many slow processes "
            f"as normal. I see the company at its best when {trait['recognition']} is noticed and practical knowledge "
            f"shapes the plan."
        )
    if question_no == 10:
        return (
            f"The single improvement I would choose is better follow-through after decisions are announced. People need "
            f"to know who owns the next step, what changed, and how success will be checked. This would address "
            f"{site_profile['friction']} without changing the whole organization again. For me, it would also support "
            f"{trait['growth']} and make {profile['need']} easier to achieve."
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
                persona = persona_label(collar, role)
                text = f"RESPONDENT: {site}, interviewee #{interviewee_no}, {persona}\nQUESTION: {question}\nANSWER: {answer}"
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
