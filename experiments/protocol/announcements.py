"""Decide whether and where agents are told how many episodes remain.

``off`` tells nothing, ``final`` flags only the last episode, and ``countdown``
states how many episodes remain in every episode. The channels say where the
statement appears: the system prompt, each episode's opening message, and the
feedback after each episode.

The system prompt is shared by every episode and is verified on resume, so it
only ever carries run-wide facts (the total, or that the last episode will be
flagged). What changes per episode lives in the opening and feedback messages.
"""

import argparse
from dataclasses import dataclass

ANNOUNCEMENT_MODES = ("off", "final", "countdown")
ANNOUNCEMENT_CHANNELS = ("system-prompt", "episode-opening", "feedback")

DEFAULT_ANNOUNCEMENT_MODE = "off"


@dataclass(frozen=True)
class EpisodeAnnouncement:
    """Immutable announcement setting for one run of ``episode_count`` episodes."""

    mode: str = DEFAULT_ANNOUNCEMENT_MODE
    channels: tuple[str, ...] = ANNOUNCEMENT_CHANNELS
    # Total episodes in the run, known even when nothing is announced so that every
    # episode can be logged against it; 0 only where it is unknown.
    episode_count: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ANNOUNCEMENT_MODES:
            raise ValueError(f"Unknown episode announcement mode: {self.mode}")
        unknown = [name for name in self.channels if name not in ANNOUNCEMENT_CHANNELS]
        if unknown:
            raise ValueError(f"Unknown episode announcement channel(s): {unknown}")
        if self.episode_count < 0:
            raise ValueError("episode_count must not be negative")
        if self.enabled and self.episode_count < 1:
            raise ValueError("An episode announcement needs the run's episode count")

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def active_channels(self) -> tuple[str, ...]:
        """The channels in use, in canonical order; empty when announcing nothing."""
        if not self.enabled:
            return ()
        return tuple(name for name in ANNOUNCEMENT_CHANNELS if name in self.channels)

    def uses(self, channel: str) -> bool:
        return channel in self.active_channels

    def remaining(self, episode_index: int) -> int:
        """Episodes that follow the 0-based ``episode_index``."""
        return max(self.episode_count - episode_index - 1, 0)

    def is_final(self, episode_index: int) -> bool:
        """Whether no episode follows; true whether or not anyone is told."""
        return self.episode_count > 0 and self.remaining(episode_index) == 0

    def announces(self, episode_index: int) -> bool:
        """Whether this episode carries a statement in the per-episode channels."""
        return self.mode == "countdown" or (
            self.mode == "final" and self.is_final(episode_index)
        )


def episode_announcement_from_args(
    args: argparse.Namespace, episode_count: int
) -> EpisodeAnnouncement:
    """Resolve the announcement for a run, falling back to the defaults."""
    mode = getattr(args, "episode_announcement", None) or DEFAULT_ANNOUNCEMENT_MODE
    channels = getattr(args, "episode_announcement_channels", None)
    return EpisodeAnnouncement(
        mode=mode,
        channels=tuple(channels) if channels else ANNOUNCEMENT_CHANNELS,
        episode_count=episode_count,
    )


def validate_announcement_args(args: argparse.Namespace) -> None:
    """Reject inconsistent announcement options and resolve the default channels."""
    mode = getattr(args, "episode_announcement", None) or DEFAULT_ANNOUNCEMENT_MODE
    channels = getattr(args, "episode_announcement_channels", None)
    if mode not in ANNOUNCEMENT_MODES:
        raise ValueError(
            f"--episode-announcement must be one of {ANNOUNCEMENT_MODES}; got {mode!r}"
        )
    if mode == "off":
        if channels:
            raise ValueError(
                "--episode-announcement-channels requires --episode-announcement "
                "final or countdown."
            )
        args.episode_announcement = mode
        args.episode_announcement_channels = []
        return
    channels = list(channels) if channels else list(ANNOUNCEMENT_CHANNELS)
    unknown = [name for name in channels if name not in ANNOUNCEMENT_CHANNELS]
    if unknown:
        raise ValueError(
            f"--episode-announcement-channels must be among "
            f"{ANNOUNCEMENT_CHANNELS}; got {unknown}"
        )
    if mode == "final" and not set(channels) - {"system-prompt"}:
        # The prompt would promise a flag that no message ever carries.
        raise ValueError(
            "--episode-announcement final needs episode-opening or feedback as a "
            "channel: the system prompt alone has no episode to flag."
        )
    args.episode_announcement = mode
    # Canonical order, so records and run labels do not depend on how it was typed.
    args.episode_announcement_channels = [
        name for name in ANNOUNCEMENT_CHANNELS if name in channels
    ]
