"""Fail-closed validation for HiGS ablation/sensitivity results."""
from __future__ import annotations

from higs_ablation_protocol import validate_ablation_protocol
from higs_paper_results import (
    HigsPaperResultError,
    validate_result as _validate_paper_result,
    validate_result_set as _validate_paper_result_set,
)


class HigsAblationResultError(HigsPaperResultError):
    """Raised when a result cannot enter the ablation matrix."""


def validate_ablation_result(result: dict, protocol: dict) -> None:
    """Validate one ablation result against the ablation protocol plan."""
    try:
        _validate_paper_result(
            result, protocol, protocol_validator=validate_ablation_protocol
        )
    except HigsPaperResultError as exc:
        raise HigsAblationResultError(str(exc)) from exc


def validate_ablation_result_set(
    results: list[dict],
    protocol: dict,
    require_complete: bool = False,
    methods: set[str] | None = None,
    hardware: set[str] | None = None,
) -> dict:
    """Validate a set of ablation results and report coverage."""
    try:
        return _validate_paper_result_set(
            results,
            protocol,
            require_complete=require_complete,
            methods=methods,
            hardware=hardware,
            protocol_validator=validate_ablation_protocol,
        )
    except HigsPaperResultError as exc:
        raise HigsAblationResultError(str(exc)) from exc
