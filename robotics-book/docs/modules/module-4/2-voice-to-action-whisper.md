---
sidebar_position: 2
title: "Voice-to-Action with Whisper"
---

# Voice-to-Action with Whisper

## Introduction to Voice Processing in Robotics

Voice commands provide a natural interface for human-robot interaction. In this chapter, we'll implement a voice recognition system using OpenAI's Whisper model to convert spoken commands into structured data that can be processed by our robotic system.

## Overview of Whisper

Whisper is a robust automatic speech recognition (ASR) system developed by OpenAI. It's particularly well-suited for robotics applications because:

- **Multilingual support**: Works with multiple languages
- **Robustness**: Handles various accents and background noise
- **Real-time capable**: Can process audio streams efficiently
- **Open-source**: Available for commercial use

## Setting Up Whisper for Robotics

### Installation

First, let's install the required dependencies:

```bash
pip install openai-whisper
pip install pyaudio  # For audio input
pip install sounddevice  # Alternative audio library
pip install transformers  # For additional NLP processing
```

### Basic Whisper Implementation

Here's a basic implementation of Whisper for voice command recognition:

```python
import whisper
import torch
import pyaudio
import wave
import io
import numpy as np
from typing import Optional

class WhisperVoiceProcessor:
    def __init__(self, model_size: str = "base"):
        """
        Initialize the Whisper voice processor.

        Args:
            model_size: Size of the Whisper model ('tiny', 'base', 'small', 'medium', 'large')
        """
        print(f"Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper works best at 16kHz
        self.chunk = 1024
        self.record_seconds = 5  # Default recording length

    def record_audio(self, duration: int = 5) -> np.ndarray:
        """
        Record audio from the microphone.

        Args:
            duration: Recording duration in seconds

        Returns:
            Audio data as numpy array
        """
        p = pyaudio.PyAudio()

        stream = p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        print(f"Recording for {duration} seconds...")
        frames = []

        for _ in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("Recording finished.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        # Normalize to [-1, 1]
        audio_np = audio_np.astype(np.float32) / 32768.0

        return audio_np

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data using Whisper.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            Transcribed text
        """
        # Ensure audio is on the correct device
        if torch.cuda.is_available():
            audio_tensor = torch.from_numpy(audio_data).to("cuda")
            self.model = self.model.to("cuda")
        else:
            audio_tensor = torch.from_numpy(audio_data)

        # Transcribe
        result = self.model.transcribe(audio_tensor)
        return result["text"].strip()

    def process_voice_command(self, timeout: int = 10) -> Optional[str]:
        """
        Process a voice command with timeout.

        Args:
            timeout: Maximum time to wait for command in seconds

        Returns:
            Transcribed command or None if timeout
        """
        try:
            audio_data = self.record_audio(duration=min(timeout, 5))
            transcription = self.transcribe_audio(audio_data)
            return transcription
        except Exception as e:
            print(f"Error processing voice command: {e}")
            return None

# Example usage
if __name__ == "__main__":
    processor = WhisperVoiceProcessor(model_size="base")

    print("Say a command...")
    command = processor.process_voice_command()

    if command:
        print(f"Recognized command: {command}")
    else:
        print("No command recognized.")
```

## Integrating Voice Recognition with ROS 2

To integrate voice recognition with ROS 2, we need to create a node that listens for voice commands and publishes them as ROS 2 messages:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import whisper
import torch
import pyaudio
import numpy as np
from threading import Thread
import time

class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__('voice_command_node')

        # Publisher for recognized commands
        self.command_publisher = self.create_publisher(String, 'voice_commands', 10)

        # Initialize Whisper processor
        self.whisper_model = whisper.load_model("base")
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024

        # Audio recording parameters
        self.record_seconds = 3
        self.is_listening = False

        # Timer for continuous listening
        self.listen_timer = self.create_timer(0.1, self.check_for_speech)

        self.get_logger().info('Voice Command Node initialized')

    def check_for_speech(self):
        """Check for speech and process commands."""
        if not self.is_listening:
            # Simple voice activity detection could be implemented here
            # For now, we'll trigger recording based on external events
            pass

    def record_and_recognize(self) -> str:
        """Record audio and recognize speech."""
        p = pyaudio.PyAudio()

        stream = p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        self.get_logger().info(f"Recording for {self.record_seconds} seconds...")
        frames = []

        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        self.get_logger().info("Recording finished.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_np = audio_np.astype(np.float32) / 32768.0

        # Transcribe using Whisper
        if torch.cuda.is_available():
            audio_tensor = torch.from_numpy(audio_np).to("cuda")
            self.whisper_model = self.whisper_model.to("cuda")
        else:
            audio_tensor = torch.from_numpy(audio_np)

        result = self.whisper_model.transcribe(audio_tensor)
        transcription = result["text"].strip()

        return transcription

    def start_listening(self):
        """Start listening for voice commands."""
        self.is_listening = True
        self.get_logger().info("Started listening for voice commands")

    def stop_listening(self):
        """Stop listening for voice commands."""
        self.is_listening = False
        self.get_logger().info("Stopped listening for voice commands")

    def process_command(self):
        """Process a voice command in a separate thread."""
        if self.is_listening:
            try:
                transcription = self.record_and_recognize()
                if transcription:
                    # Publish the recognized command
                    msg = String()
                    msg.data = transcription
                    self.command_publisher.publish(msg)
                    self.get_logger().info(f'Published command: "{transcription}"')
            except Exception as e:
                self.get_logger().error(f'Error processing voice command: {e}')

def main(args=None):
    rclpy.init(args=args)

    voice_node = VoiceCommandNode()

    # Start a separate thread for processing commands
    def command_processor():
        while rclpy.ok():
            voice_node.process_command()
            time.sleep(0.5)  # Small delay between recordings

    processor_thread = Thread(target=command_processor, daemon=True)
    processor_thread.start()

    try:
        rclpy.spin(voice_node)
    except KeyboardInterrupt:
        pass
    finally:
        voice_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Voice Command Processing Pipeline

The voice command processing pipeline consists of several stages:

1. **Audio Capture**: Recording audio from the microphone
2. **Preprocessing**: Converting audio to the format required by Whisper
3. **Transcription**: Converting speech to text using Whisper
4. **Command Parsing**: Extracting actionable commands from transcribed text
5. **ROS 2 Publishing**: Publishing commands as ROS 2 messages

## Advanced Voice Processing Features

### Voice Activity Detection (VAD)

To improve efficiency, we can implement voice activity detection to only process audio when speech is detected:

```python
import webrtcvad
import collections

class VoiceActivityDetector:
    def __init__(self, aggressiveness=3):
        """
        Initialize Voice Activity Detector.

        Args:
            aggressiveness: VAD sensitivity (0-3, higher = more sensitive)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.rate * self.frame_duration / 1000) * 2  # 2 bytes per sample

    def is_speech(self, audio_frame):
        """
        Check if the audio frame contains speech.

        Args:
            audio_frame: Audio data as bytes

        Returns:
            True if speech is detected, False otherwise
        """
        try:
            return self.vad.is_speech(audio_frame, self.rate)
        except:
            return False
```

### Command Validation and Error Handling

```python
class CommandValidator:
    def __init__(self):
        # Define valid command patterns
        self.valid_actions = [
            "move", "go", "navigate", "pick", "grasp", "place",
            "clean", "organize", "fetch", "bring", "take"
        ]

        self.valid_locations = [
            "kitchen", "living room", "bedroom", "bathroom",
            "office", "dining room", "hallway"
        ]

    def validate_command(self, command_text):
        """
        Validate if the command makes sense for the robot.

        Args:
            command_text: The transcribed command

        Returns:
            Tuple of (is_valid, processed_command, confidence)
        """
        command_lower = command_text.lower()

        # Check for valid action words
        has_action = any(action in command_lower for action in self.valid_actions)

        # Check for valid location words
        has_location = any(location in command_lower for location in self.valid_locations)

        # Simple confidence scoring
        confidence = 0.0
        if has_action:
            confidence += 0.3
        if has_location:
            confidence += 0.2

        # Check for action-object patterns
        if any(word in command_lower for word in ["to", "the", "a", "an"]):
            confidence += 0.2

        return confidence > 0.5, command_text, confidence
```

## Real-World Considerations

### Audio Quality Optimization

For better performance in real-world environments:

1. **Noise Reduction**: Use noise reduction techniques to improve audio quality
2. **Microphone Placement**: Position microphones optimally for clear voice capture
3. **Echo Cancellation**: Implement echo cancellation for speaker feedback
4. **Beamforming**: Use microphone arrays for directional audio capture

### Performance Optimization

```python
class OptimizedWhisperProcessor:
    def __init__(self):
        # Load model once and keep it in memory
        self.model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")

        # Use smaller models for faster inference if accuracy allows
        # Consider using quantized models for edge deployment
        self.is_warm = False

    def warm_up(self):
        """Warm up the model with a dummy input."""
        if not self.is_warm:
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            dummy_tensor = torch.from_numpy(dummy_audio)
            _ = self.model.transcribe(dummy_tensor)
            self.is_warm = True

    def transcribe_optimized(self, audio_data):
        """Optimized transcription with warm-up."""
        self.warm_up()
        audio_tensor = torch.from_numpy(audio_data)

        # Use faster inference settings if needed
        result = self.model.transcribe(
            audio_tensor,
            fp16=torch.cuda.is_available(),  # Use fp16 on GPU for faster inference
            temperature=0.0  # Use greedy decoding for consistency
        )

        return result["text"].strip()
```

## Integration with LLM Planning

The voice commands processed by Whisper will be passed to the LLM planning system, which we'll cover in the next chapter. For now, let's see how to structure the output for downstream processing:

```python
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class VoiceCommand:
    """Structured representation of a voice command."""
    original_text: str
    processed_text: str
    action: str
    target_object: Optional[str] = None
    target_location: Optional[str] = None
    confidence: float = 0.0
    timestamp: float = 0.0

def parse_voice_command(transcription: str) -> VoiceCommand:
    """
    Parse a transcribed voice command into structured format.

    Args:
        transcription: The raw transcribed text

    Returns:
        VoiceCommand object with parsed information
    """
    # Simple parsing logic (in practice, you might use more sophisticated NLP)
    import re

    original_text = transcription
    processed_text = transcription.lower().strip()

    # Extract action (simple keyword matching)
    action = ""
    for keyword in ["move", "go", "pick", "grasp", "place", "clean", "fetch"]:
        if keyword in processed_text:
            action = keyword
            break

    # Extract target object (simple pattern matching)
    target_object = None
    object_patterns = [
        r"pick up the (\w+)",
        r"grasp the (\w+)",
        r"take the (\w+)",
        r"bring me the (\w+)"
    ]

    for pattern in object_patterns:
        match = re.search(pattern, processed_text)
        if match:
            target_object = match.group(1)
            break

    # Extract target location
    target_location = None
    location_patterns = [
        r"to the (\w+)",
        r"in the (\w+)",
        r"from the (\w+)"
    ]

    for pattern in location_patterns:
        match = re.search(pattern, processed_text)
        if match:
            target_location = match.group(1)
            break

    return VoiceCommand(
        original_text=original_text,
        processed_text=processed_text,
        action=action,
        target_object=target_object,
        target_location=target_location,
        confidence=0.8,  # Placeholder - in real system this would come from validation
        timestamp=time.time()
    )
```

## Summary

In this chapter, we've explored how to implement voice recognition using Whisper for robotics applications. We've covered:

- Setting up Whisper for voice command recognition
- Integrating with ROS 2 for message publishing
- Advanced features like voice activity detection
- Performance optimization techniques
- Structuring voice commands for downstream processing

The voice recognition system we've built serves as the foundation for the LLM planning pipeline we'll implement in the next chapter. The transcribed commands will be processed by our language model to generate executable robotic actions.

In the next chapter, we'll explore how to convert these natural language commands into executable ROS 2 actions using Large Language Models.