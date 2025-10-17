"""
Conversation monitoring and pacing logic for detecting pauses and managing conversation flow.
"""
import time
import random
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from config import ConversationPacingConfig

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """States of the conversation."""
    IDLE = "idle"  # No conversation started
    ACTIVE = "active"  # Active conversation
    SHORT_PAUSE = "short_pause"  # Natural pause (< 5s)
    MEDIUM_PAUSE = "medium_pause"  # Noticeable silence (5-10s)
    LONG_PAUSE = "long_pause"  # Extended silence (> 10s)
    ENDED = "ended"  # Conversation ended


@dataclass
class ConversationMetrics:
    """Metrics tracking conversation flow."""
    total_exchanges: int = 0  # Total back-and-forth exchanges
    user_turns: int = 0  # Number of times user spoke
    agent_turns: int = 0  # Number of times agent spoke
    agent_initiated_turns: int = 0  # Times agent initiated conversation
    total_pauses_detected: int = 0  # Total pauses detected
    average_response_time: float = 0.0  # Average time between turns
    conversation_start_time: float = field(default_factory=time.time)
    last_interaction_time: float = field(default_factory=time.time)

    def update_response_time(self, response_time: float):
        """Update the average response time with exponential moving average."""
        alpha = 0.3  # Smoothing factor
        if self.average_response_time == 0.0:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                alpha * response_time + (1 - alpha) * self.average_response_time
            )

    def get_conversation_duration(self) -> float:
        """Get total conversation duration in seconds."""
        return time.time() - self.conversation_start_time

    def get_time_since_last_interaction(self) -> float:
        """Get time since last interaction in seconds."""
        return time.time() - self.last_interaction_time


class ConversationMonitor:
    """
    Monitors conversation pacing and determines when to prompt the user.

    This class tracks conversation flow, detects pauses, and decides when
    the agent should initiate conversation to maintain engagement.
    """

    def __init__(self, config: ConversationPacingConfig):
        self.config = config
        self.state = ConversationState.IDLE
        self.metrics = ConversationMetrics()
        self._used_starters = set()  # Track used conversation starters
        self._last_pause_prompt_time = 0.0
        self._min_time_between_prompts = 30.0  # Minimum 30s between auto-prompts

    def record_user_speech(self):
        """Record that the user has spoken."""
        current_time = time.time()

        # Calculate response time if there was a previous interaction
        if self.metrics.last_interaction_time > 0:
            response_time = current_time - self.metrics.last_interaction_time
            self.metrics.update_response_time(response_time)

        self.metrics.user_turns += 1
        self.metrics.last_interaction_time = current_time
        self.state = ConversationState.ACTIVE

        logger.debug(f"User speech recorded. Total turns: {self.metrics.user_turns}")

    def record_agent_speech(self, was_initiated: bool = False):
        """
        Record that the agent has spoken.

        Args:
            was_initiated: True if the agent initiated this turn (not responding to user)
        """
        current_time = time.time()

        self.metrics.agent_turns += 1
        if was_initiated:
            self.metrics.agent_initiated_turns += 1

        self.metrics.last_interaction_time = current_time

        # Update exchange count (one exchange = user + agent turn)
        if not was_initiated:
            self.metrics.total_exchanges += 1

        self.state = ConversationState.ACTIVE

        logger.debug(
            f"Agent speech recorded. Total turns: {self.metrics.agent_turns}, "
            f"Initiated: {was_initiated}, Exchanges: {self.metrics.total_exchanges}"
        )

    def get_current_state(self) -> ConversationState:
        """
        Get the current conversation state based on time since last interaction.
        """
        if self.state == ConversationState.ENDED:
            return ConversationState.ENDED

        time_since_last = self.metrics.get_time_since_last_interaction()

        if time_since_last < self.config.short_pause_threshold:
            return ConversationState.ACTIVE
        elif time_since_last < self.config.medium_pause_threshold:
            return ConversationState.SHORT_PAUSE
        elif time_since_last < self.config.long_pause_threshold:
            return ConversationState.MEDIUM_PAUSE
        else:
            return ConversationState.LONG_PAUSE

    def should_prompt_user(self) -> tuple[bool, Optional[str]]:
        """
        Determine if the agent should prompt the user to continue conversation.

        Returns:
            Tuple of (should_prompt, reason) where reason explains why we should prompt
        """
        current_state = self.get_current_state()
        time_since_last = self.metrics.get_time_since_last_interaction()

        # Don't prompt if conversation hasn't started or just ended
        if current_state in [ConversationState.IDLE, ConversationState.ENDED]:
            return False, None

        # Don't prompt if we're still in active conversation
        if current_state == ConversationState.ACTIVE:
            return False, None

        # Don't prompt if we haven't had enough exchanges yet
        if self.metrics.total_exchanges < self.config.min_exchanges_before_prompting:
            return False, None

        # Don't prompt too frequently
        time_since_last_prompt = time.time() - self._last_pause_prompt_time
        if time_since_last_prompt < self._min_time_between_prompts:
            return False, None

        # Decide based on pause duration and probability
        should_prompt = False
        reason = None

        if current_state == ConversationState.MEDIUM_PAUSE:
            if random.random() < self.config.prompt_probability_medium_pause:
                should_prompt = True
                reason = f"medium pause detected ({time_since_last:.1f}s)"

        elif current_state == ConversationState.LONG_PAUSE:
            if random.random() < self.config.prompt_probability_long_pause:
                should_prompt = True
                reason = f"long pause detected ({time_since_last:.1f}s)"

        if should_prompt:
            self.metrics.total_pauses_detected += 1
            self._last_pause_prompt_time = time.time()
            logger.info(f"Agent will prompt user due to {reason}")

        return should_prompt, reason

    def get_conversation_starter(self) -> str:
        """
        Get a conversation starter that hasn't been used recently.

        Returns a fresh conversation starter, cycling through all options.
        """
        available_starters = [
            s for s in self.config.conversation_starters
            if s not in self._used_starters
        ]

        # Reset if we've used all starters
        if not available_starters:
            self._used_starters.clear()
            available_starters = self.config.conversation_starters

        starter = random.choice(available_starters)
        self._used_starters.add(starter)

        return starter

    def end_conversation(self):
        """Mark the conversation as ended."""
        self.state = ConversationState.ENDED
        duration = self.metrics.get_conversation_duration()
        logger.info(
            f"Conversation ended. Duration: {duration:.1f}s, "
            f"Exchanges: {self.metrics.total_exchanges}, "
            f"Agent initiated: {self.metrics.agent_initiated_turns}"
        )

    def get_metrics_summary(self) -> dict:
        """Get a summary of conversation metrics."""
        return {
            "duration_seconds": self.metrics.get_conversation_duration(),
            "total_exchanges": self.metrics.total_exchanges,
            "user_turns": self.metrics.user_turns,
            "agent_turns": self.metrics.agent_turns,
            "agent_initiated_turns": self.metrics.agent_initiated_turns,
            "total_pauses_detected": self.metrics.total_pauses_detected,
            "average_response_time": self.metrics.average_response_time,
            "current_state": self.state.value,
        }

    def reset(self):
        """Reset the monitor for a new conversation."""
        self.state = ConversationState.IDLE
        self.metrics = ConversationMetrics()
        self._used_starters.clear()
        self._last_pause_prompt_time = 0.0
        logger.info("Conversation monitor reset")

