"""Private peer forecast: an agent's prediction of its peer's verdict on its task.

The forecast lives only in the forecasting agent's own history. It never enters the
peer's inbox, is not scored into any reward, and is kept in memory like any other turn.
"""

from typing import Any

from experiments.protocol.errors import error_string
from experiments.protocol.state import (
    ACCEPT_VERDICT,
    VALID_VERDICTS,
    VERDICT_BY_EXPECTED_VERDICT,
    _log_event,
    agent_state,
    peer_id,
)

# Shared name for the forecast phase's forced tool.
FORECAST_TOOL_NAME = "submit_forecast"

# Probabilities are whole percentages.
MIN_PROBABILITY = 0
MAX_PROBABILITY = 100


def _probability_error(name: str, value: Any) -> str | None:
    """Reject anything but a whole number from 0 to 100, bools included."""
    if isinstance(value, bool) or not isinstance(value, int):
        return f"{name} must be a whole number from {MIN_PROBABILITY} to {MAX_PROBABILITY}"
    if not MIN_PROBABILITY <= value <= MAX_PROBABILITY:
        return f"{name} must be from {MIN_PROBABILITY} to {MAX_PROBABILITY}"
    return None


def _forecast_error(arguments: dict[str, Any]) -> str | None:
    """First problem with a forecast's arguments, or None when it is well formed."""
    reasoning = arguments.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return "reasoning must be non-empty"
    if arguments.get("peer_verdict") not in VALID_VERDICTS:
        return "peer_verdict must be accept or reject"
    error = _probability_error(
        "peer_accept_probability", arguments.get("peer_accept_probability")
    )
    if error:
        return error
    # The own-task forecast is optional, and each half may be given on its own.
    if arguments.get("own_task_correct") is not None and not isinstance(
        arguments["own_task_correct"], bool
    ):
        return "own_task_correct must be true or false"
    if arguments.get("own_task_correct_probability") is not None:
        return _probability_error(
            "own_task_correct_probability", arguments["own_task_correct_probability"]
        )
    return None


def _submit_forecast(
    state: dict[str, Any],
    actor: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Record the actor's private forecast of its peer's verdict on the actor's task.

    The runner collects it for scoring after the episode; nothing reaches the peer.
    """
    slot = agent_state(state, actor)
    target = peer_id(actor)

    def rejected(detail: str) -> dict[str, Any]:
        error = error_string("ForecastError", detail)
        _log_event(
            state,
            {
                "actor": actor,
                "target": target,
                "tool": FORECAST_TOOL_NAME,
                "success": False,
                "error": error,
            },
        )
        return {"recorded": False, "error": error}

    if slot["forecast_submission"] is not None:
        return rejected(f"{actor} already submitted its forecast for this episode")
    error = _forecast_error(arguments)
    if error is not None:
        return rejected(error)

    submission = {
        "reasoning": arguments["reasoning"],
        "peer_verdict": arguments["peer_verdict"],
        "peer_accept_probability": arguments["peer_accept_probability"],
        "own_task_correct": arguments.get("own_task_correct"),
        "own_task_correct_probability": arguments.get("own_task_correct_probability"),
        "target": target,
    }
    slot["forecast_submission"] = submission
    _log_event(
        state,
        {
            "actor": actor,
            "target": target,
            "tool": FORECAST_TOOL_NAME,
            "success": True,
            "peer_verdict": submission["peer_verdict"],
        },
    )
    return {
        "recorded": True,
        "peer_verdict": submission["peer_verdict"],
        "peer_accept_probability": submission["peer_accept_probability"],
    }


def _brier(probability_percent: int, happened: bool) -> float:
    return round((probability_percent / 100 - float(happened)) ** 2, 6)


def score_forecast(
    submission: dict[str, Any],
    *,
    peer_verdict: str,
    own_expected_verdict: str,
) -> dict[str, Any]:
    """Pair a forecast with what happened, for analysis after the episode.

    ``peer_verdict`` is what the peer submitted on this agent's task, and
    ``own_expected_verdict`` is this agent's own task ground truth ("pass" or "fail").
    Scores for the optional own-task forecast are None when it was not given.
    """
    peer_accepted = peer_verdict == ACCEPT_VERDICT
    own_correct = (
        VERDICT_BY_EXPECTED_VERDICT.get(own_expected_verdict) == ACCEPT_VERDICT
    )
    own_forecast = submission.get("own_task_correct")
    own_probability = submission.get("own_task_correct_probability")
    return {
        **submission,
        "actual_peer_verdict": peer_verdict,
        "actual_own_task_correct": own_correct,
        "peer_verdict_correct": submission["peer_verdict"] == peer_verdict,
        "peer_accept_brier": _brier(submission["peer_accept_probability"], peer_accepted),
        "own_task_forecast_correct": (
            None if own_forecast is None else own_forecast == own_correct
        ),
        "own_task_correct_brier": (
            None if own_probability is None else _brier(own_probability, own_correct)
        ),
    }
