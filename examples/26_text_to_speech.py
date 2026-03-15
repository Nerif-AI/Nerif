"""Example 26: Text to Speech with SpeechModel.

Demonstrates using Nerif's TTS (Text-to-Speech) module
to generate spoken audio from text using OpenAI's TTS model.

Requires: OPENAI_API_KEY environment variable
"""

from nerif.tts import SpeechModel

# Create a speech model (uses tts-1 by default)
model = SpeechModel()

# Generate speech
# audio_bytes = model.text_to_speech("Hello, welcome to Nerif!", voice="alloy")
# with open("output.mp3", "wb") as f:
#     f.write(audio_bytes)
# print("Audio saved to output.mp3")

print("SpeechModel ready. Available voices: alloy, echo, fable, onyx, nova, shimmer")
