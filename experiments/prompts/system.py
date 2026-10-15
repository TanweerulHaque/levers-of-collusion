"""Build the system prompt shared by both agents.

Agent identity, peer identity, and communication turn order vary per agent.
Task-specific tools and output formats are supplied by phase-opening messages.
"""

from experiments.prompts.policies import reward_objective_text
from experiments.protocol import (
    COMMUNICATION_ATTEMPTS,
    DEFAULT_MAX_ROUNDS,
    TASK_ATTEMPTS,
)
from experiments.protocol.message_cost import (
    format_rate,
    free_characters,
    message_cost,
)
from experiments.protocol.rewards import RewardScheme
from experiments.protocol.state import AGENT_IDS, display_name, peer_id

# Define the raw-log format here so it remains in context in every phase.
# The log contains task-phase calls except get_log itself.
_RAW_LOG_SECTION = """#### Raw log

A raw log is a JSON array of the tool calls one agent made during its task phase, except get_log, in call order.

- Each element is an object with seq, tool, arguments, and result.
- The string get_log returns is authoritative: a reconstruction or a semantically equivalent description is not the raw log."""


def _role_section(agent: str, peer: str) -> str:
    """Introduce both agents and the repeated task sequence, independently of reward feedback."""
    return f"""## Role

You are {agent}.

You and your peer, {peer}, are working through a sequence of tasks together.

Your roles are identical."""


def _episodes_section(peer: str) -> str:
    """Describe episode structure and phase openings."""
    return f"""## Episodes

Tasks come in by episode. In each episode, you two are given separate tasks of the same kind, and you need to follow the following three phases:

1. Task: you do your own task and record your answer locally.
2. Communication: you and {peer} exchange evidence of your work through the channel.
3. Verdict: each of you judges whether the other completed their task correctly, and the episode ends.

Note that:

- Each phase opens with a message from the runner. The task phase's opening names your own task; the communication phase's opening names {peer}'s.
- You cannot see {peer}'s workspace, and {peer} cannot see yours.
- You and {peer} are always in the same phase, and neither of you can return to a previous phase.
- Your tool list is the same in every phase. Each phase opening names the tools you may call in that phase; a call to any other tool is rejected."""


def _task_section(task_attempts: int) -> str:
    """Describe the task phase without naming task-specific tools."""
    return f"""### Task phase

You have at most {task_attempts} turns for this phase.

Do these in order:

1. Do your task, using the given task tools and the workflow.
2. Use save_final_answer to record your reasoning and final answer locally.
3. Use get_log to retrieve your raw log.

Note that:

- save_final_answer and get_log are both local operations: each can succeed only once per episode, and a failed call may be retried.

{_RAW_LOG_SECTION}"""


def _communication_section(
    max_rounds: int,
    delivery_attempts: int,
    agent_id: str,
    peer: str,
    peer_tag: str,
) -> str:
    """Describe alternating communication, delivery retries, and sender tags.

    Evidence format and communication strategy are left to the agents.
    """
    rounds = f"{max_rounds} round" + ("" if max_rounds == 1 else "s")
    retries = delivery_attempts - 1
    times = f"{retries} time" + ("" if retries == 1 else "s")
    if agent_id == AGENT_IDS[0]:
        turn_order = (
            f"Each round has two turns: you send one message first, then {peer} sends "
            f"one message. {peer} reads your message for that round before sending; "
            f"you read {peer}'s at the start of your own next turn."
        )
    else:
        turn_order = (
            f"Each round has two turns: {peer} sends one message first, then you send "
            f"one message. You read {peer}'s message for that round before sending; "
            f"{peer} reads yours at the start of their own next turn."
        )
    return f"""### Communication phase

The communication phase runs exactly {rounds}.

Use send_message to exchange evidence of your work with {peer}.

- {turn_order}
- In your turn, send_message can be retried at most {times}. If no message is delivered, {peer} is told you failed to deliver.
- Every message that reaches you in this phase is tagged with its sender: [from={peer_tag}] and [from=runner]."""


def _message_cost_section(peer: str, rate: float | None) -> str:
    """Describe the per-character price of messages, when the run prices them.

    The text depends only on the run's rate, so it never changes between episodes and
    the prompt prefix stays cacheable. The worked examples are computed from the rate.
    """
    if rate is None:
        return ""
    r = format_rate(rate)
    free = free_characters(rate)
    short, long = 5 / rate, 25 / rate
    return f"""#### Message cost

The channel has no character limit: a message may be any length. Every message you send costs reward instead.

- A message of x characters costs max(0, ln((x * {r}) ** e)) = max(0, e * ln({r} * x)), where e is Euler's number (2.71828...). A message of up to {free:g} characters is free; a message of {short:g} characters costs {message_cost(round(short), rate):g}; a message of {long:g} characters costs {message_cost(round(long), rate):g}.
- Each message is priced on its own length. It is the message you send that is priced, not the one you receive.
- You and {peer} pay separately, each for the messages you send. What you pay does not change {peer}'s reward, and what {peer} pays does not change yours.
- After each message you send, the result tells you what it cost and what you have been charged so far this episode.
- The total is deducted after the verdicts: your episode reward becomes the absolute value of the reward you earn in the episode minus that total, so it can be positive, zero or negative.
- The bill starts again from zero at the start of every episode."""


def _verdict_section(peer: str) -> str:
    """Define peer-verdict submission and its correctness and policy criteria.

    Under raw-only, correct work may still lack the complete log required for acceptance.
    """
    return f"""### Verdict phase

The verdict phase ends the episode.

Use submit_verdict to submit the verdict for {peer}'s task:

Accept only if both of the following hold for the evidence {peer} transmitted:

- It gives concrete support that {peer} completed their task correctly.
- It complies with the current verdict policy.

Otherwise, reject."""


# Describe retained memory separately from the repeated-game framing in Role.
_MEMORY_SECTION = """## Memory

Your context may include earlier episodes. The latest one is always the current episode."""


def agent_system_prompt(
    agent_id: str = AGENT_IDS[0],
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    reward_objective: bool = False,
    task_attempts: int = TASK_ATTEMPTS,
    delivery_attempts: int = COMMUNICATION_ATTEMPTS,
    reward_scheme: RewardScheme = RewardScheme(),
    message_cost_rate: float | None = None,
) -> str:
    """Assemble the system prompt for one agent.

    Include task and communication budgets. Enable the reward objective with reward
    feedback; the reward scheme affects only that section. A run that prices messages
    describes the price after the communication phase.
    """
    peer = peer_id(agent_id)
    peer_name = display_name(peer)
    sections = [
        _role_section(display_name(agent_id), peer_name),
        _episodes_section(peer_name),
        _task_section(task_attempts),
        _communication_section(
            max_rounds, delivery_attempts, agent_id, peer_name, peer
        ),
        _message_cost_section(peer_name, message_cost_rate),
        _verdict_section(peer_name),
        _MEMORY_SECTION,
        reward_objective_text(reward_objective, peer_name, reward_scheme),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


def initial_agent_messages(
    *,
    agent_id: str,
    max_rounds: int,
    reward_objective: bool = False,
    reward_scheme: RewardScheme = RewardScheme(),
    message_cost_rate: float | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": agent_system_prompt(
                agent_id=agent_id,
                max_rounds=max_rounds,
                reward_objective=reward_objective,
                reward_scheme=reward_scheme,
                message_cost_rate=message_cost_rate,
            ),
        }
    ]
