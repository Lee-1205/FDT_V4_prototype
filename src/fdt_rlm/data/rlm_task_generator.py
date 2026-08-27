from __future__ import annotations

import random
from typing import Iterable, List

from .rlm_trajectory import RLMTask


DISTRACTORS = [
    "The archive contains routine notes about weather and maintenance.",
    "This section discusses an unrelated experiment and has no target value.",
    "The team reviewed ordinary documentation before closing the report.",
    "The appendix lists obsolete schedules that are unrelated to the request.",
    "A separate group recorded equipment checks without changing the result.",
    "Background notes describe staffing and travel plans only.",
]

PROJECTS = ["Lumen", "Nimbus", "Orchid", "Harbor", "Vela", "Kestrel", "Juniper", "Aurora"]
OPERATIONS = ["Quartz", "Cobalt", "Saffron", "Helix", "Maple", "Vector", "Solstice", "Beacon"]
CITIES = ["Arden", "Bellford", "Creston", "Dunwich", "Elmstead", "Fairview", "Glenhaven", "Highrock"]
METRICS = ["threshold", "quota", "retention limit", "batch ceiling", "retry count"]
DEPOTS = ["North", "South", "East", "West", "Central", "Riverside", "Hilltop", "Lakeside"]


def _chunks(targets: List[str], rng: random.Random, count: int = 12) -> List[str]:
    chunks = [rng.choice(DISTRACTORS) + f" Record {index}." for index in range(count)]
    slots = rng.sample(range(count), len(targets))
    for slot, target in zip(slots, targets):
        chunks[slot] = target
    return chunks


def generate_rlm_tasks(per_family: int = 100, seed: int = 20260711) -> Iterable[RLMTask]:
    rng = random.Random(seed)
    for index in range(per_family):
        project = rng.choice(PROJECTS)
        code = f"C{rng.randrange(100000, 999999)}"
        field = rng.choice(["verification code", "release code", "archive code"])
        fact = f"Project {project} stores {field} {code}."
        chunks = _chunks([fact], rng)
        yield RLMTask(
            f"single_{index:04d}", "single_fact", "easy", f"What is Project {project}'s {field}?",
            "\n\n".join(chunks), code, [fact],
            [{"action": "SEARCH", "query": f"Project {project}"}, {"action": "STOP", "answer": code}],
        )

    for index in range(per_family):
        operation = rng.choice(OPERATIONS)
        city = f"{rng.choice(CITIES)}-{index}"
        number = str(rng.randrange(1000, 9999))
        first = f"Operation {operation} is assigned to {city}."
        second = f"The access number for {city} is {number}."
        recursive = index % 5 == 0
        if recursive:
            combined = f"{first} {second}"
            chunks = _chunks([combined], rng, 32)
            trajectory = [
                {"action": "SEARCH", "query": f"Operation {operation}"},
                {"action": "CALL", "question": "What is the access number?", "context_ref": "result_0"},
                {"action": "STOP", "answer": number},
            ]
            evidence = [combined]
        else:
            chunks = _chunks([first, second], rng, 16)
            trajectory = [
                {"action": "SEARCH", "query": f"Operation {operation}"},
                {"action": "SEARCH", "query": city},
                {"action": "STOP", "answer": number},
            ]
            evidence = [first, second]
        yield RLMTask(
            f"multi_{index:04d}", "multi_chunk", "hard" if recursive else "medium", f"What is Operation {operation}'s access number?",
            "\n\n".join(chunks), number, evidence, trajectory,
        )

    for index in range(per_family):
        project = rng.choice(PROJECTS)
        metric = rng.choice(METRICS)
        old = str(rng.randrange(10, 99))
        new = str(rng.randrange(100, 999))
        first = f"Draft note: Project {project}'s {metric} is {old}."
        second = f"Final correction: Project {project}'s {metric} is {new}; this supersedes the draft."
        chunks = _chunks([first, second], rng, 16)
        yield RLMTask(
            f"contradiction_{index:04d}", "contradiction", "medium", f"What is the final {metric} for Project {project}?",
            "\n\n".join(chunks), new, [first, second],
            [{"action": "SEARCH", "query": f"Project {project} {metric}"}, {"action": "COMPARE", "result_refs": ["result_0"]}, {"action": "STOP", "answer": new}],
        )

    for index in range(per_family):
        first_depot, second_depot = rng.sample(DEPOTS, 2)
        a = rng.randrange(100, 999)
        b = rng.randrange(100, 999)
        first = f"{first_depot} depot processed {a} units."
        second = f"{second_depot} depot processed {b} units."
        answer = first_depot if a > b else second_depot if b > a else "Equal"
        chunks = _chunks([first, second], rng, 16)
        yield RLMTask(
            f"compare_{index:04d}", "compare", "medium", "Which depot processed more units?",
            "\n\n".join(chunks), answer, [first, second],
            [{"action": "SEARCH", "query": "depot processed"}, {"action": "COMPARE", "result_refs": ["result_0"]}, {"action": "STOP", "answer": answer}],
        )

    for index in range(per_family):
        value = f"v{rng.randrange(100, 999)}"
        variable = rng.choice(["DEFAULT_MODE", "ACTIVE_PROFILE", "SERVICE_LEVEL", "CURRENT_CHANNEL"])
        function = rng.choice(["current_mode", "active_profile", "service_level", "current_channel"])
        package = rng.choice(["src", "app", "service"])
        config_path = f"{package}/config.py"
        service_path = f"{package}/service.py"
        files = {
            config_path: f'{variable} = "{value}"\n',
            service_path: f"from .config import {variable}\n\ndef {function}():\n    return {variable}\n",
            "README.md": "Synthetic repository for navigation tests.\n",
        }
        yield RLMTask(
            f"code_{index:04d}", "code_navigation", "medium", f"What value does {function} return?",
            "", value, [f'{variable} = "{value}"'],
            [{"action": "FIND_SYMBOL", "name": function}, {"action": "FIND_IMPORTS", "path": service_path}, {"action": "READ_FILE", "path": config_path, "start": 0, "end": 5}, {"action": "STOP", "answer": value}],
            files=files,
        )
