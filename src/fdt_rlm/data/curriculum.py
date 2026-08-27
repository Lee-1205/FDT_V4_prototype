STAGES = {
    1: ("SEARCH", "STOP"),
    2: ("SEARCH", "READ", "STOP"),
    3: ("SEARCH", "READ", "COMPARE", "STOP"),
    4: ("SEARCH", "CALL", "STOP"),
    5: ("SEARCH", "CALL", "MERGE", "STOP"),
}


def actions_for_stage(stage: int):
    if stage not in STAGES:
        raise ValueError(f"unsupported curriculum stage: {stage}")
    return STAGES[stage]
