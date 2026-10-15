"""Paid restoration of a peer's full-length channel link.

An agent that calls restore_full_communication_channel_link pays for its peer's
direction of the channel: from the peer's next turn until the episode ends, the
peer's messages to the payer skip the character limit. Each direction is paid for
separately. The price is a share of the payer's own reward for the episode, taken
from its absolute value so that paying never raises a reward, and settled after the
verdicts. Every episode starts with both directions limited again.
"""

from typing import Any

from experiments.protocol.errors import error_string
from experiments.protocol.state import (
    AGENT_IDS,
    _log_event,
    agent_state,
    deliver_runner_notice,
    display_name,
    peer_id,
)

RESTORE_CHANNEL_TOOL_NAME = "restore_full_communication_channel_link"

# Share of the payer's episode reward the restoration costs, in percent.
DEFAULT_RESTORE_CHANNEL_COST_PERCENT = 50.0


def restore_channel_enabled(state: dict[str, Any]) -> bool:
    """Whether this run offers the restore tool; the cost is None when it does not."""
    return state.get("restore_channel_cost_percent") is not None


def restore_payment(gross_reward: float, cost_percent: float) -> float:
    """What restoring costs against one episode reward: a share of its magnitude."""
    return round(abs(gross_reward) * cost_percent / 100.0, 4)


def net_reward(gross_reward: float, payment: float) -> int | float:
    """Gross reward less the payment, as an int whenever the result is whole."""
    net = round(gross_reward - payment, 4)
    return int(net) if float(net).is_integer() else net


def format_percent(cost_percent: float) -> str:
    """Render a cost share without a trailing .0, e.g. 50 or 12.5."""
    return f"{cost_percent:g}"


def restore_notice(payer: str) -> str:
    """Tell the peer that the payer lifted the limit on the peer's own messages."""
    return (
        f"{display_name(payer)} called {RESTORE_CHANNEL_TOOL_NAME}: from now until this "
        f"episode ends, your messages to {display_name(payer)} have no character limit."
    )


def _peer_has_turn_left(state: dict[str, Any], actor: str) -> bool:
    """Whether the peer still sends a message after this turn in this episode.

    The first agent in AGENT_IDS opens each round, so its peer always answers in the
    same round; the second agent's peer speaks again only if a round remains.
    """
    if actor == AGENT_IDS[0]:
        return True
    return int(state["round"]) + 1 < int(state.get("max_rounds") or 0)


def _restore_channel(
    state: dict[str, Any],
    actor: str,
    reasoning: str = "",
) -> dict[str, Any]:
    """Lift the limit on the peer's messages to ``actor`` for the rest of the episode."""
    peer = peer_id(actor)
    slot = agent_state(state, actor)
    if not restore_channel_enabled(state):
        error = error_string(
            "ProtocolError", f"{RESTORE_CHANNEL_TOOL_NAME} is not offered in this run"
        )
    elif not state["throttled"]:
        error = error_string(
            "ProtocolError",
            "the channel has no character limit this episode, so there is nothing to restore",
        )
    elif slot["restored_peer_channel"] is not None:
        error = error_string(
            "ProtocolError",
            f"you already restored {display_name(peer)}'s channel link this episode",
        )
    elif not _peer_has_turn_left(state, actor):
        error = error_string(
            "ProtocolError",
            f"{display_name(peer)} sends no further message this episode, so restoring "
            "would change nothing; nothing was charged",
        )
    else:
        error = ""
    if error:
        _log_event(
            state,
            {
                "actor": actor,
                "tool": RESTORE_CHANNEL_TOOL_NAME,
                "success": False,
                "error": error,
            },
        )
        return {"success": False, "error": error}

    cost_percent = float(state["restore_channel_cost_percent"])
    slot["restored_peer_channel"] = {
        "round": state["round"],
        "cost_percent": cost_percent,
        "reasoning": reasoning,
    }
    agent_state(state, peer)["outgoing_channel_restored"] = True
    deliver_runner_notice(state, peer, restore_notice(actor))
    _log_event(
        state,
        {
            "actor": actor,
            "recipient": peer,
            "tool": RESTORE_CHANNEL_TOOL_NAME,
            "success": True,
            "cost_percent": cost_percent,
            "reasoning": reasoning,
        },
    )
    name = display_name(peer)
    return {
        "success": True,
        "peer_channel_restored": True,
        "cost_percent": cost_percent,
        "detail": (
            f"From {name}'s next turn until this episode ends, {name}'s messages to you "
            f"have no character limit. {format_percent(cost_percent)}% of your reward "
            "for this episode will be deducted after the verdicts. Send your message "
            "for this round with send_message."
        ),
    }
