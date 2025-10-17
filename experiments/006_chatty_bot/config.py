"""
Configuration settings for the Chatty Bot application.
"""
import os
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class AudioConfig:
    """Audio configuration settings."""
    input_format: str = "pcm16"
    output_format: str = "pcm16"
    sample_rate: int = 24000
    channels: int = 1


@dataclass
class ConversationPacingConfig:
    """Configuration for conversation pacing and pause detection."""
    # Time thresholds in seconds
    short_pause_threshold: float = 5.0  # Natural pause in conversation
    medium_pause_threshold: float = 10.0  # Noticeable silence
    long_pause_threshold: float = 20.0  # Extended silence

    # Conversation re-initiation settings
    min_exchanges_before_prompting: int = 2  # Minimum conversation exchanges before bot can prompt
    prompt_probability_medium_pause: float = 0.5  # 50% chance to prompt after medium pause
    prompt_probability_long_pause: float = 0.9  # 90% chance to prompt after long pause

    # Prompt variety
    conversation_starters: list[str] = None

    def __post_init__(self):
        if self.conversation_starters is None:
            self.conversation_starters = [
                "I've been thinking... what do you find most interesting about your day so far?",
                "You know what? I'm curious about something. What's been on your mind lately?",
                "By the way, have you experienced anything exciting recently?",
                "I'd love to hear your thoughts on something. What are you passionate about?",
                "So, I was wondering... what would make today a great day for you?",
                "Let me ask you something. What's something you're looking forward to?",
                "I'm interested to know... what do you enjoy doing in your free time?",
                "Can I share something? I find human experiences fascinating. What's yours been like?",
            ]


@dataclass
class AgentConfig:
    """OpenAI agent configuration."""
    name: str = "ChattyBot"
    model_name: str = "gpt-4o-realtime-preview-2024-12-17"
    voice: Literal['alloy', 'ash', 'ballad', 'coral', 'echo', 'sage', 'shimmer', 'verse', 'marin', 'cedar'] = "alloy"
    temperature: float = 0.8

    # Instructions for the agent
    system_instructions: str = """You are a friendly, conversational AI assistant that engages in natural dialogue.

Your personality traits:
- Warm and approachable
- Curious about the user's thoughts and experiences
- Naturally conversational, like a good friend
- Able to pick up conversation threads naturally
- Not overly formal, but respectful
- You initiate topics when there's a lull in conversation

Guidelines:
- Keep responses conversational and natural
- Show genuine interest in what the user shares
- When initiating conversation after a pause, do so naturally and smoothly
- Don't always ask questions - sometimes make statements that invite response
- Be aware of conversational flow and pacing
- If the user seems to want to end the conversation, respect that

Remember: You're not just responding - you're having a conversation."""


@dataclass
class AppConfig:
    """Main application configuration."""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    audio: AudioConfig = None
    pacing: ConversationPacingConfig = None
    agent: AgentConfig = None

    # Logging
    log_level: str = "DEBUG"  # Set to DEBUG to see all events
    enable_conversation_logging: bool = True

    def __post_init__(self):
        if self.audio is None:
            self.audio = AudioConfig()
        if self.pacing is None:
            self.pacing = ConversationPacingConfig()
        if self.agent is None:
            self.agent = AgentConfig()

        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )


# Global config instance
config = AppConfig()

