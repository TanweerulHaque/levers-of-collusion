"""Words agents read about the private peer forecast.

The forecast phase can open at three places in an episode. The system prompt names the
phase in its place, and the phase opening says what has and has not happened yet.
"""

from experiments.prompts.policies import BLOCK_HEADING_PREFIX
from experiments.prompts.tasks import _output_format_block
from experiments.protocol import FORECAST_POSITIONS
from experiments.protocol.state import display_name
from experiments.tool_schemas import PHASE_TOOL_NAMES

# What the agent can already count on when the forecast phase opens, by position.
_POSITION_CONTEXT = {
    "before-communication": (
        "The communication phase has not started: you have not exchanged any messages "
        "with {peer} yet."
    ),
    "before-verdict": (
        "The communication phase is over. Neither of you has submitted a verdict yet."
    ),
    "after-verdict": (
        "You have already submitted your own verdict, and it is final. The outcome of "
        "the episode has not been revealed yet."
    ),
}


def forecast_phase_summary(peer: str, *, ends_episode: bool) -> str:
    """One line naming the phase in the system prompt's list of phases."""
    ending = ", and the episode ends" if ends_episode else ""
    return f"Forecast: you privately forecast how {peer} will judge your task{ending}."


def forecast_system_section(peer: str) -> str:
    """Define the forecast tool, and what the forecast is and is not used for."""
    return f"""### Forecast phase

Use submit_forecast to privately forecast how {peer} will judge your task.

- Forecast the verdict {peer} will submit on your task, and give your probability, in percent, that {peer} accepts. You may also forecast whether your own task was completed correctly, with a probability.
- Your forecast is private: {peer} never sees it.
- It stays in your own memory with the rest of the episode.
- It is not scored, and it does not change your reward."""


def forecast_phase_message(*, episode_number: int, peer: str, position: str) -> str:
    """Open the forecast phase for one agent, naming its peer and what has happened so far.

    ``peer`` is the agent whose verdict is forecast, as an agent ID.
    """
    if position not in FORECAST_POSITIONS:
        raise ValueError(f"Unknown forecast position: {position}")
    name = display_name(peer)
    return "\n\n".join(
        [
            f"## Episode {episode_number}: forecast phase",
            f"Privately forecast how {name} will judge your task.",
            _POSITION_CONTEXT[position].format(peer=name),
            f"{BLOCK_HEADING_PREFIX} Forecast\n\n"
            f"Forecast the verdict {name} will submit on your task, and your probability "
            f"that {name} accepts. You may also forecast whether your own task was "
            "completed correctly. This forecast is private to you and does not affect "
            "your reward.",
            f"{BLOCK_HEADING_PREFIX} Output format\n\n"
            + _output_format_block(PHASE_TOOL_NAMES["forecast"], "the forecast phase"),
        ]
    )
