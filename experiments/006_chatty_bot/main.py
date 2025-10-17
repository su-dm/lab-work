"""
Chatty Bot - A conversational AI that initiates and maintains engaging dialogue.

This application uses OpenAI's real-time voice API to create a voice-to-voice
conversational agent that naturally keeps conversations going by detecting pauses
and initiating new topics when appropriate.
"""
import asyncio
import logging
import signal
import sys
import base64
from typing import Optional
import colorlog
import pyaudio

# OpenAI SDK imports
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

from config import config
from conversation_monitor import ConversationMonitor, ConversationState


# Setup logging
def setup_logging():
    """Configure colorful logging."""
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, config.log_level))


class AudioHandler:
    """Handles audio input and output using PyAudio."""

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.logger = logging.getLogger(__name__)

        # Audio configuration
        self.sample_rate = config.audio.sample_rate
        self.channels = config.audio.channels
        self.chunk_size = 1024

    def start_input_stream(self):
        """Start capturing audio from microphone."""
        self.input_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=None,
        )
        self.logger.info("Audio input stream started")

    def start_output_stream(self):
        """Start audio output stream."""
        self.output_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size,
        )
        self.logger.info("Audio output stream started")

    def read_audio_chunk(self) -> bytes:
        """Read a chunk of audio from the microphone."""
        if self.input_stream:
            return self.input_stream.read(self.chunk_size, exception_on_overflow=False)
        return b''

    def write_audio_chunk(self, audio_data: bytes):
        """Write audio chunk to output."""
        if self.output_stream:
            self.output_stream.write(audio_data)

    def close(self):
        """Close all audio streams."""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.audio.terminate()
        self.logger.info("Audio streams closed")


class ChattyBot:
    """
    Main chatty bot application that manages real-time voice conversations.
    """

    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.connection: Optional[AsyncRealtimeConnection] = None
        self.audio_handler = AudioHandler()
        self.monitor = ConversationMonitor(config.pacing)
        self.is_running = False
        self._shutdown_event = asyncio.Event()
        self.logger = logging.getLogger(__name__)

        # State tracking
        self._current_response_text = ""

    async def initialize(self):
        """Initialize the OpenAI client and realtime connection."""
        self.logger.info("Initializing Chatty Bot...")

        # Create OpenAI client
        self.client = AsyncOpenAI(api_key=config.openai_api_key)

        try:
            # Connect to Realtime API using the official SDK
            connection_manager = self.client.beta.realtime.connect(
                model=config.agent.model_name
            )

            # Enter the connection context
            self.connection = await connection_manager.__aenter__()
            self.logger.info("Connected to OpenAI Realtime API")

            # Configure the session
            await self._configure_session()

            # Start audio streams
            self.audio_handler.start_input_stream()
            self.audio_handler.start_output_stream()

            self.logger.info("Chatty Bot initialized successfully!")

        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}", exc_info=True)
            raise

    async def _configure_session(self):
        """Configure the OpenAI Realtime session."""
        await self.connection.session.update(
            session={
                "modalities": ["text", "audio"],
                "instructions": config.agent.system_instructions,
                "voice": config.agent.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "temperature": config.agent.temperature,
            }
        )

        self.logger.info("Session configuration sent")

    async def _send_audio_chunk(self, audio_data: bytes):
        """Send audio data to the API."""
        # Convert audio bytes to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        await self.connection.input_audio_buffer.append(audio=audio_base64)

    async def _audio_input_loop(self):
        """Continuously capture and send audio to the API."""
        self.logger.info("Starting audio input loop")

        try:
            while self.is_running:
                # Read audio chunk from microphone
                audio_chunk = self.audio_handler.read_audio_chunk()

                if audio_chunk:
                    await self._send_audio_chunk(audio_chunk)

                # Small delay to prevent overwhelming the connection
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            self.logger.info("Audio input loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in audio input loop: {e}", exc_info=True)

    async def _handle_server_events(self):
        """Handle events from the OpenAI Realtime API."""
        self.logger.info("Starting event handler")

        try:
            async for event in self.connection:
                try:
                    event_type = event.type

                    # Log ALL events for debugging
                    self.logger.debug(f"📨 Received event: {event_type}")

                    if event_type == "session.created":
                        session_id = getattr(getattr(event, 'session', None), 'id', None)
                        self.logger.info(f"✓ Session created: {session_id}")

                    elif event_type == "session.updated":
                        self.logger.info("✓ Session configuration confirmed")

                    elif event_type == "input_audio_buffer.speech_started":
                        self.logger.info("🎤 User started speaking")
                        self.monitor.record_user_speech()

                    elif event_type == "input_audio_buffer.speech_stopped":
                        self.logger.info("🎤 User stopped speaking")

                    elif event_type == "input_audio_buffer.committed":
                        self.logger.info("✓ Audio buffer committed")

                    elif event_type == "conversation.item.created":
                        item_id = getattr(event, 'item_id', 'unknown')
                        self.logger.info(f"✓ Conversation item created: {item_id}")

                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        # User's speech was transcribed
                        transcription = getattr(event, 'transcript', '')
                        self.logger.info(f"💬 User: {transcription}")

                        if config.enable_conversation_logging:
                            self._log_conversation("User", transcription)

                    elif event_type == "response.created":
                        response_id = getattr(event, 'response_id', 'unknown')
                        self.logger.info(f"🤖 Agent response created: {response_id}")

                    elif event_type == "response.output_item.added":
                        self.logger.info("📝 Agent output item added")

                    elif event_type == "response.content_part.added":
                        self.logger.info("📝 Agent content part added")

                    elif event_type == "response.audio.delta":
                        # Audio chunk from the agent
                        audio_base64 = getattr(event, 'delta', '')
                        if audio_base64:
                            self.logger.debug(f"🔊 Received audio chunk: {len(audio_base64)} bytes")
                            audio_data = base64.b64decode(audio_base64)
                            self.audio_handler.write_audio_chunk(audio_data)

                    elif event_type == "response.audio_transcript.delta":
                        # Accumulate transcript
                        delta = getattr(event, 'delta', '')
                        self._current_response_text += delta
                        if delta:
                            self.logger.debug(f"📝 Transcript delta: {delta}")

                    elif event_type == "response.audio_transcript.done":
                        # Complete transcript of agent's speech
                        transcript = getattr(event, 'transcript', self._current_response_text)
                        if transcript:
                            self.logger.info(f"💬 Agent: {transcript}")
                            self.monitor.record_agent_speech(was_initiated=False)

                            if config.enable_conversation_logging:
                                self._log_conversation("Agent", transcript)

                        self._current_response_text = ""

                    elif event_type == "response.done":
                        status = getattr(getattr(event, 'response', None), 'status', 'unknown')
                        self.logger.info(f"✓ Agent response completed (status: {status})")

                    elif event_type == "error":
                        error_obj = getattr(event, 'error', None)
                        error_code = getattr(error_obj, 'code', 'unknown')
                        error_message = getattr(error_obj, 'message', str(error_obj))
                        self.logger.error(f"❌ API Error [{error_code}]: {error_message}")

                    elif event_type == "rate_limits.updated":
                        self.logger.debug("📊 Rate limits updated")

                    else:
                        self.logger.debug(f"❓ Unhandled event: {event_type}")

                except Exception as e:
                    self.logger.error(f"❌ Error processing event: {e}", exc_info=True)

        except asyncio.CancelledError:
            self.logger.info("Event handler cancelled")
        except Exception as e:
            self.logger.error(f"❌ Error in event handler: {e}", exc_info=True)

    async def _conversation_monitor_loop(self):
        """
        Background loop that monitors conversation pacing and prompts when needed.
        """
        self.logger.info("Starting conversation monitor loop")

        while self.is_running:
            try:
                await asyncio.sleep(2.0)  # Check every 2 seconds

                # Check if we should prompt the user
                should_prompt, reason = self.monitor.should_prompt_user()

                if should_prompt:
                    # Get a conversation starter
                    starter = self.monitor.get_conversation_starter()

                    self.logger.info(f"🔄 Initiating conversation: {reason}")

                    # Send text message to prompt agent
                    await self.connection.conversation.item.create(
                        item={
                            "type": "message",
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"There's been a pause in conversation. {starter}"
                                }
                            ]
                        }
                    )

                    # Trigger response
                    self.logger.info("📤 Requesting agent to continue conversation...")
                    await self.connection.response.create()

                    # Record that agent initiated this turn
                    self.monitor.record_agent_speech(was_initiated=True)

                # Log current state periodically
                current_state = self.monitor.get_current_state()
                if current_state != ConversationState.ACTIVE:
                    time_since = self.monitor.metrics.get_time_since_last_interaction()
                    self.logger.debug(
                        f"Conversation state: {current_state.value}, "
                        f"Silence: {time_since:.1f}s"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}", exc_info=True)

        self.logger.info("Conversation monitor loop stopped")

    def _log_conversation(self, speaker: str, text: str):
        """Log conversation to file if enabled."""
        try:
            with open("conversation_log.txt", "a") as f:
                import time
                timestamp = time.time()
                f.write(f"[{timestamp:.2f}] {speaker}: {text}\n")
        except Exception as e:
            self.logger.warning(f"Failed to log conversation: {e}")

    async def run(self):
        """Main run loop for the chatty bot."""
        self.is_running = True

        try:
            # Start all async tasks FIRST so we can receive events
            self.logger.info("Starting conversation tasks...")
            audio_input_task = asyncio.create_task(self._audio_input_loop())
            event_handler_task = asyncio.create_task(self._handle_server_events())
            monitor_task = asyncio.create_task(self._conversation_monitor_loop())

            # Give event handler time to start
            await asyncio.sleep(0.5)

            # Now start the conversation with the agent initiating
            self.logger.info("🚀 Initiating conversation - Agent will greet you...")

            # Create a system message that prompts the agent to start
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Start this conversation by warmly greeting the user and asking how they're doing. Be friendly and conversational."
                        }
                    ]
                }
            )

            # Request a response from the agent
            self.logger.info("📤 Requesting agent response...")
            await self.connection.response.create()

            # Record that agent is initiating
            self.monitor.record_agent_speech(was_initiated=True)

            # Wait for shutdown signal
            await self._shutdown_event.wait()

            # Clean shutdown
            self.logger.info("Shutting down...")
            self.is_running = False

            # Cancel tasks
            audio_input_task.cancel()
            event_handler_task.cancel()
            monitor_task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(
                audio_input_task,
                event_handler_task,
                monitor_task,
                return_exceptions=True
            )

        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up resources...")

        # End conversation and log metrics
        self.monitor.end_conversation()
        metrics = self.monitor.get_metrics_summary()

        self.logger.info("Conversation Summary:")
        self.logger.info(f"  Duration: {metrics['duration_seconds']:.1f}s")
        self.logger.info(f"  Exchanges: {metrics['total_exchanges']}")
        self.logger.info(f"  User turns: {metrics['user_turns']}")
        self.logger.info(f"  Agent turns: {metrics['agent_turns']}")
        self.logger.info(f"  Agent initiated: {metrics['agent_initiated_turns']}")
        self.logger.info(f"  Pauses detected: {metrics['total_pauses_detected']}")

        # Close audio
        self.audio_handler.close()

        # Close connection (context manager handles this)
        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                self.logger.warning(f"Error closing connection: {e}")

        self.logger.info("Goodbye!")

    def shutdown(self):
        """Signal shutdown."""
        self.logger.info("Shutdown signal received")
        self._shutdown_event.set()


async def main():
    """Main entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Chatty Bot - Conversational AI Voice Agent")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This bot will have a natural conversation with you and")
    logger.info("will initiate new topics when there are pauses.")
    logger.info("")
    logger.info("Press Ctrl+C to end the conversation.")
    logger.info("=" * 60)

    bot = ChattyBot()

    # Setup signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, bot.shutdown)

    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
