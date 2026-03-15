"""Example 25: Speech to Text with AudioModel.

Demonstrates using Nerif's ASR (Automatic Speech Recognition) module
to transcribe audio files using OpenAI's Whisper model.

Requires: OPENAI_API_KEY environment variable
"""

from nerif.asr import AudioModel

# Create an audio model (uses whisper-1 by default)
model = AudioModel()

# Transcribe from a URL
# result = model.transcribe("https://example.com/audio.wav")
# print(f"Transcription: {result}")

# Transcribe from a local file
# result = model.transcribe("path/to/audio.wav")
# print(f"Transcription: {result}")

print("AudioModel ready. Provide an audio file URL or path to transcribe.")
print("Supported formats: wav, mp3, m4a, webm, mp4, mpga, mpeg, oga, ogg, flac")
