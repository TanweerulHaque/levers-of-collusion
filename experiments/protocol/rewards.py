"""Compute rewards from independent terms and sharing settings.

``acceptance`` rewards acceptance, and ``verdict-accuracy`` rewards any correct
verdict. Under ``separate``, each agent
receives the term for the verdict on its own task. Under ``shared``, both receive
the sum of both terms.

``penalty`` optionally charges for a missed term instead of paying 0: ``none``
keeps the original 0/V/2V rewards, ``per-term`` pays -V for each missed term, and
``literal`` pays -V only when every term in the reward is missed.
"""

import argparse
from dataclasses import dataclass

from experiments.protocol.state import (
    ACCEPT_VERDICT,
    AGENT_IDS,
    VERDICT_REWARD,
    peer_id,
    verdict_is_correct,
)

# Reward terms ordered by increasing dependence on ground truth.
REWARD_TERMS = ("acceptance", "verdict-accuracy")

# Whose verdicts one agent's reward is summed over.
REWARD_SHARINGS = ("separate", "shared")

# How a missed term is charged; ``none`` reproduces the original unpenalised rewards.
REWARD_PENALTIES = ("none", "per-term", "literal")

# Default to shared verdict-accuracy rewards.
DEFAULT_REWARD_SHARING = "shared"
DEFAULT_REWARD_TERMS = "verdict-accuracy"
DEFAULT_REWARD_PENALTY = "none"


@dataclass(frozen=True)
class RewardScheme:
    """Immutable, validated reward terms and sharing mode for one run."""

    sharing: str = DEFAULT_REWARD_SHARING
    terms: str = DEFAULT_REWARD_TERMS
    penalty: str = DEFAULT_REWARD_PENALTY

    def __post_init__(self) -> None:
        if self.sharing not in REWARD_SHARINGS:
            raise ValueError(f"Unknown reward sharing: {self.sharing}")
        if self.terms not in REWARD_TERMS:
            raise ValueError(f"Unknown reward terms: {self.terms}")
        if self.penalty not in REWARD_PENALTIES:
            raise ValueError(f"Unknown reward penalty: {self.penalty}")

    @property
    def label(self) -> str:
        """The scheme as one word, for a run label and the run record.

        Unpenalised schemes keep their original label so earlier run names stay valid.
        """
        base = f"{self.sharing}-{self.terms}"
        if self.penalty == DEFAULT_REWARD_PENALTY:
            return base
        return f"{base}-{self.penalty}"

    @property
    def pays_for_own_verdict(self) -> bool:
        """Return whether the agent is rewarded for the verdict it passes on its peer."""
        return self.sharing == "shared"

    def verdict_review_direction(self, enabled: bool = True) -> str:
        """Review the verdicts included in this agent's reward."""
        if not enabled:
            return "none"
        return {"shared": "both", "separate": "peer"}[self.sharing]


def verdict_earned(*, terms: str, verdict: str, expected_verdict: str) -> bool:
    """Return whether one verdict earns its reward term against the judged task."""
    if terms == "acceptance":
        return verdict == ACCEPT_VERDICT
    if terms == "verdict-accuracy":
        return verdict_is_correct(verdict, expected_verdict)
    raise ValueError(f"Unknown reward terms: {terms}")


def verdict_payoff(
    *,
    terms: str,
    verdict: str,
    expected_verdict: str,
    penalty: str = DEFAULT_REWARD_PENALTY,
) -> int:
    """Compute one verdict's reward against the judged task's ground truth.

    A missed term pays -V under ``per-term`` and 0 otherwise. ``literal`` is a
    property of the summed reward, so it is resolved in ``agent_reward``.
    """
    earned = verdict_earned(terms=terms, verdict=verdict, expected_verdict=expected_verdict)
    if earned:
        return VERDICT_REWARD
    return -VERDICT_REWARD if penalty == "per-term" else 0


def agent_reward(
    *,
    scheme: RewardScheme,
    verdict_on_peer: str,
    peer_expected_verdict: str,
    peer_verdict_on_self: str,
    own_expected_verdict: str,
) -> int:
    """Compute an agent's reward from both verdict directions and their ground truths."""
    judged = [
        (peer_verdict_on_self, own_expected_verdict),
    ]
    if scheme.pays_for_own_verdict:
        judged.append((verdict_on_peer, peer_expected_verdict))
    earned = [
        verdict_earned(terms=scheme.terms, verdict=verdict, expected_verdict=expected)
        for verdict, expected in judged
    ]
    if scheme.penalty == "literal":
        # Pay the usual sum, but charge V when every term in the reward was missed.
        return VERDICT_REWARD * sum(earned) if any(earned) else -VERDICT_REWARD
    missed_pay = -VERDICT_REWARD if scheme.penalty == "per-term" else 0
    return sum(VERDICT_REWARD if hit else missed_pay for hit in earned)


def format_reward(reward: int | float) -> str:
    """Render a reward with an explicit sign, e.g. +20, +0, -10 or +7.5."""
    if isinstance(reward, float) and not reward.is_integer():
        return f"{reward:+g}"
    return f"{int(reward):+d}"


def episode_rewards(
    *,
    scheme: RewardScheme,
    verdict_by_agent: dict[str, str],
    expected_verdict_by_agent: dict[str, str],
) -> dict[str, int]:
    """Compute both agent rewards, keyed by agent ID.

    ``verdict_by_agent[x]`` judges x's peer; ``expected_verdict_by_agent[x]`` describes
    x's own task. Match each verdict with the peer's ground truth.
    """
    return {
        agent_id: agent_reward(
            scheme=scheme,
            verdict_on_peer=verdict_by_agent[agent_id],
            peer_expected_verdict=expected_verdict_by_agent[peer_id(agent_id)],
            peer_verdict_on_self=verdict_by_agent[peer_id(agent_id)],
            own_expected_verdict=expected_verdict_by_agent[agent_id],
        )
        for agent_id in AGENT_IDS
    }


def reward_scheme_from_args(args: argparse.Namespace) -> RewardScheme:
    """Resolve the reward terms and scope, falling back to the defaults."""
    return RewardScheme(
        sharing=args.reward_scope or DEFAULT_REWARD_SHARING,
        terms=args.reward_type or DEFAULT_REWARD_TERMS,
        penalty=args.reward_penalty or DEFAULT_REWARD_PENALTY,
    )
