"""Codabench entry point: are_robust(model_id, reasoning_effort, problems) -> list[bool]."""
from fragility import predict


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    return predict(model_id, reasoning_effort, problems)
