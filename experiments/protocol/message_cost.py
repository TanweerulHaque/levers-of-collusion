"""Per-character pricing of the communication channel.

In a ``--message-cost`` run the channel is not throttled: a message may be any length.
Instead every delivered message costs its sender reward, priced by its own length

    cost(x) = max(0, ln((x * rate) ** e))  =  max(0, e * ln(x * rate))

where ``x`` is the number of characters in the message (one message per agent per
round, so one communication round for one agent), ``rate`` is ``--message-cost-rate``
(default 0.005) and ``e`` is Euler's number. A message of up to ``1 / rate`` characters
is free. Each agent pays for its own messages; the two bills are never combined.

The bill is settled after the verdicts, from the absolute value of the verdict reward
the agent earned in that episode: ``net = |gross| - total_cost``. Nothing is clipped, so
a net reward can be positive, zero or negative. Every episode starts a new bill.
"""

import argparse
import math
from typing import Any

# Price constant of the experiment: ``--message-cost-rate`` overrides it.
DEFAULT_MESSAGE_COST_RATE = 0.005


def message_cost_enabled(state: dict[str, Any]) -> bool:
    """Whether this run prices messages; the rate is None when it does not."""
    return state.get("message_cost_rate") is not None


def message_cost(chars: int, rate: float) -> float:
    """What one message of ``chars`` characters costs: max(0, ln((chars * rate) ** e)).

    ``ln(y ** e) = e * ln(y)``, so this is computed in that form; ``y`` below 1 (a
    message shorter than ``1 / rate``) has a negative logarithm and costs nothing.
    """
    if chars <= 0 or rate <= 0:
        return 0.0
    return round(max(0.0, math.e * math.log(chars * rate)), 4)


def free_characters(rate: float) -> float:
    """The longest message that costs nothing: the length where ``chars * rate`` is 1."""
    return 1.0 / rate


def counted_reward(gross_reward: float) -> int | float:
    """The reward the bill is taken from: the absolute value of the verdict reward."""
    return abs(gross_reward)


def net_reward_after_cost(gross_reward: float, cost: float) -> int | float:
    """``|gross| - cost``, as an int whenever the result is whole. No lower bound."""
    net = round(counted_reward(gross_reward) - cost, 4)
    return int(net) if float(net).is_integer() else net


def format_rate(rate: float) -> str:
    """Render the rate without a trailing .0, e.g. 0.005 or 0.01."""
    return f"{rate:g}"


def rate_slug(rate: float) -> str:
    """The rate as a path-safe word: 0.005 becomes 0_005, so folder names hold no dot."""
    return format_rate(rate).replace(".", "_")


def message_cost_rate_from_args(args: argparse.Namespace) -> float | None:
    """The run's price per character, or None when messages are not priced."""
    if not getattr(args, "message_cost", False):
        return None
    return float(args.message_cost_rate)
