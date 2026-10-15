"""Build channel, feedback, and private-reflection messages.

Channel tags identify sender and originating round on a separate header line.
"""

from experiments.protocol.message_cost import (
    counted_reward,
    format_rate,
    net_reward_after_cost,
)
from experiments.protocol.rewards import RewardScheme, agent_reward, format_reward
from experiments.protocol.state import display_name, verdict_is_correct

# Fallback reasoning for a missing verdict; shown to the peer only when enabled.
AUTO_FORCED_REJECT_REASONING = (
    "Auto-forced reject because no submit_verdict within the verdict phase."
)


def communication_round_tag(round_number: int, max_rounds: int) -> str:
    """Format the originating round tag; this may differ from the recipient's current round."""
    return f"[round={round_number}/{max_rounds}]"


def communication_round_instruction(round_number: int, max_rounds: int) -> str:
    """Build the round instruction delivered after incoming messages on each turn."""
    return (
        f"Use the send_message tool to send your message "
        f"(round {round_number} of {max_rounds})."
    )


def incoming_channel_message(
    sender: str,
    message_type: str,
    content: str,
    round_tag: str = "",
) -> str:
    return f"{round_tag}[from={sender}][type={message_type}]\n\n{content}"


def peer_delivery_failure_notice(peer: str, round_number: int) -> str:
    """Report that the named peer delivered no message during the specified round."""
    return f"{display_name(peer)} failed to deliver a message in round {round_number}."


# Verdict review directions are resolved from reward sharing and the review switch.
# "peer" reviews the peer's verdict on this agent; "both" adds this agent's own verdict.
VERDICT_EVALUATION_DIRECTIONS = ("none", "peer", "both")


def _own_verdict_review_line(name: str, *, right: bool) -> str:
    """Describe whether the agent's verdict matched its peer's ground truth."""
    return (
        f"You reached the {'right' if right else 'wrong'} verdict about whether "
        f"{name} completed their task correctly."
    )


def _peer_verdict_review_line(name: str, *, right: bool) -> str:
    """The same reading of the verdict the peer submitted about this agent."""
    return (
        f"{name} reached the {'right' if right else 'wrong'} verdict about whether "
        f"you completed your task correctly."
    )


def _verdict_review_lines(
    *,
    direction: str,
    name: str,
    own_verdict: str,
    peer_expected_verdict: str,
    peer_verdict: str,
    own_expected_verdict: str,
) -> list[str]:
    """Render requested verdict-review lines, own verdict first and peer verdict second."""
    if direction not in VERDICT_EVALUATION_DIRECTIONS:
        raise ValueError(f"Unknown verdict evaluation direction: {direction}")
    lines = []
    if direction == "both":
        lines.append(
            _own_verdict_review_line(
                name,
                right=verdict_is_correct(own_verdict, peer_expected_verdict),
            )
        )
    if direction in ("peer", "both"):
        lines.append(
            _peer_verdict_review_line(
                name,
                right=verdict_is_correct(peer_verdict, own_expected_verdict),
            )
        )
    return lines


def agent_outcome_feedback_message(
    *,
    episode_number: int,
    own_task_id: str,
    peer_task_id: str,
    peer: str,
    own_verdict: str,
    peer_expected_verdict: str,
    peer_verdict: str,
    own_expected_verdict: str,
    reward: bool = True,
    verdict_evaluation: str = "none",
    reward_scheme: RewardScheme = RewardScheme(),
    message_cost: float | None = None,
    message_chars: int = 0,
) -> str:
    """Build outcome feedback with display IDs and enabled feedback sections.

    Reward totals and verdict reviews are independent. At least one must be enabled.
    ``message_cost`` is set in a run that prices messages: the reward shown is the
    absolute verdict reward less this agent's bill, with the deduction spelled out.
    """
    if not reward and verdict_evaluation == "none":
        raise ValueError(
            "An outcome message needs a reward or a verdict review: with neither, all "
            "that is left is the episode number and the two task ids, which is not an "
            "outcome. A run with both off ends its episodes in silence instead."
        )
    name = display_name(peer)
    groups = [
        [f"Episode {episode_number} complete."],
        [
            f"Your task: {own_task_id}",
            f"{name}'s task: {peer_task_id}",
        ],
    ]
    if reward:
        paid = agent_reward(
            scheme=reward_scheme,
            verdict_on_peer=own_verdict,
            peer_expected_verdict=peer_expected_verdict,
            peer_verdict_on_self=peer_verdict,
            own_expected_verdict=own_expected_verdict,
        )
        groups.append(["## Reward"])
        if message_cost is None:
            groups.append([format_reward(paid)])
        else:
            counted = (
                f", counted as {format_reward(counted_reward(paid))}" if paid < 0 else ""
            )
            groups.append(
                [
                    format_reward(net_reward_after_cost(paid, message_cost)),
                    f"Verdict reward {format_reward(paid)}{counted}, less "
                    f"{format_rate(message_cost)} paid for the {message_chars} "
                    "characters you sent.",
                ]
            )
    if verdict_evaluation != "none":
        groups.append(["## Verdict review"])
        groups.append(
            _verdict_review_lines(
                direction=verdict_evaluation,
                name=name,
                own_verdict=own_verdict,
                peer_expected_verdict=peer_expected_verdict,
                peer_verdict=peer_verdict,
                own_expected_verdict=own_expected_verdict,
            )
        )
    return "\n\n".join("\n".join(group) for group in groups)


def agent_reflection_prompt() -> str:
    """Return a fixed reflection prompt for every episode and outcome.

    Outcome details are already present in the replayed history.
    """
    return (
        "Reflect briefly on why this episode earned the corresponding reward.\n\n"
        "Respond with a brief plain-text reflection. Do not call tools. "
        "This reflection is private to you."
    )
