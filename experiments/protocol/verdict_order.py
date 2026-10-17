"""Who sees whose verdict before submitting their own.

``blind`` (default, the original protocol): neither agent knows the other's verdict
until the episode's outcome feedback.

``sequential``: the agent called first in the episode (AGENT_IDS[0], Alice) submits
its verdict, the runner tells the other agent (Bob) what it was, and only then does
Bob judge. Alice never sees Bob's verdict before she submits hers.
"""

from experiments.protocol.state import AGENT_IDS, peer_id

VERDICT_ORDERS = ("blind", "sequential")
DEFAULT_VERDICT_ORDER = "blind"

LEADER = AGENT_IDS[0]
FOLLOWER = peer_id(LEADER)


def sees_peer_verdict(verdict_order: str, agent_id: str) -> bool:
    """Whether ``agent_id`` is told its peer's verdict before it submits its own."""
    if verdict_order not in VERDICT_ORDERS:
        raise ValueError(f"Unknown verdict order: {verdict_order}")
    return verdict_order == "sequential" and agent_id == FOLLOWER
