# Nerif Audio Model

Nerif provides a simple and powerful AudioModel class for audio processing tasks, particularly audio transcription using OpenAI's Whisper model.

## Overview

The AudioModel class enables you to:
- Transcribe audio files to text using state-of-the-art speech recognition
- Support multiple audio formats (mp3, mp4, mpeg, mpga, m4a, wav, webm)
- Process audio files up to 25MB in size

## Installation

The AudioModel is included in the base Nerif package:

```bash
pip install nerif
```

## Quick Start

```python
from nerif.model import AudioModel
from pathlib import Path

# Initialize the audio model
audio_model = AudioModel()

# Transcribe an audio file
transcription = audio_model.transcribe(Path("interview.mp3"))
print(transcription.text)
```

## API Reference

### AudioModel Class

A simple agent for audio tasks including transcription and speech-to-text conversion.

#### Initialization

```python
audio_model = AudioModel()
```

The AudioModel automatically initializes an OpenAI client using your environment credentials.

#### Methods

##### transcribe(file: Path)

Transcribes an audio file to text using the Whisper-1 model.

**Parameters:**
- `file` (Path): Path to the audio file to transcribe

**Returns:**
- Transcription object containing:
  - `text`: The transcribed text
  - Additional metadata from the transcription

**Example:**
```python
from pathlib import Path

# Transcribe a single file
result = audio_model.transcribe(Path("podcast.mp3"))
print(f"Transcription: {result.text}")
```

## Supported Audio Formats

The AudioModel supports the following audio formats:
- mp3
- mp4
- mpeg
- mpga
- m4a
- wav
- webm

## File Size Limitations

- Maximum file size: 25MB
- For larger files, consider splitting them into smaller segments

## Examples

### Basic Transcription

```python
from nerif.model import AudioModel
from pathlib import Path

# Initialize model
audio_model = AudioModel()

# Transcribe audio
audio_file = Path("meeting_recording.mp3")
transcription = audio_model.transcribe(audio_file)

print(f"Transcribed text: {transcription.text}")
```

### Batch Transcription

```python
from nerif.model import AudioModel
from pathlib import Path
import os

audio_model = AudioModel()

# Process multiple audio files
audio_dir = Path("audio_files")
for audio_file in audio_dir.glob("*.mp3"):
    try:
        result = audio_model.transcribe(audio_file)
        
        # Save transcription to text file
        output_file = audio_file.with_suffix('.txt')
        output_file.write_text(result.text)
        
        print(f"Transcribed {audio_file.name}")
    except Exception as e:
        print(f"Error transcribing {audio_file.name}: {e}")
```

### Transcription with Error Handling

```python
from nerif.model import AudioModel
from pathlib import Path

audio_model = AudioModel()

def safe_transcribe(file_path):
    """Safely transcribe audio with error handling"""
    try:
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            return f"Error: File {file_path} not found"
        
        # Check file size (25MB limit)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 25:
            return f"Error: File too large ({file_size_mb:.1f}MB). Maximum is 25MB"
        
        # Transcribe
        result = audio_model.transcribe(path)
        return result.text
        
    except Exception as e:
        return f"Transcription error: {str(e)}"

# Use the safe transcription function
transcription = safe_transcribe("interview.mp3")
print(transcription)
```

### Integration with Text Processing

```python
from nerif.model import AudioModel, SimpleChatModel
from pathlib import Path

# Initialize models
audio_model = AudioModel()
chat_model = SimpleChatModel()

# Transcribe audio
audio_file = Path("customer_call.mp3")
transcription = audio_model.transcribe(audio_file)

# Analyze transcription with LLM
analysis_prompt = f"""
Analyze this customer call transcription and provide:
1. Main topics discussed
2. Customer sentiment
3. Action items

Transcription:
{transcription.text}
"""

analysis = chat_model.chat(analysis_prompt)
print(analysis)
```

### Creating Subtitles

```python
from nerif.model import AudioModel
from pathlib import Path
import textwrap

audio_model = AudioModel()

def create_subtitle_chunks(text, chunk_size=50):
    """Split text into subtitle-sized chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > chunk_size:
            chunks.append(' '.join(current_chunk[:-1]))
            current_chunk = [word]
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Transcribe and create subtitles
audio_file = Path("video_audio.mp3")
transcription = audio_model.transcribe(audio_file)

# Generate subtitle chunks
subtitles = create_subtitle_chunks(transcription.text)

# Save as subtitle file (simple format)
with open("subtitles.txt", "w") as f:
    for i, subtitle in enumerate(subtitles, 1):
        f.write(f"{i}. {subtitle}\n")

print(f"Created {len(subtitles)} subtitle chunks")
```

## Best Practices

1. **File Format**: Use WAV or MP3 for best compatibility
2. **Audio Quality**: Ensure clear audio with minimal background noise
3. **File Size**: Keep files under 25MB; split longer recordings if needed
4. **Error Handling**: Always implement error handling for file operations
5. **Batch Processing**: Process multiple files sequentially to avoid rate limits

## Common Issues and Solutions

### File Too Large
```python
# Split large audio files before transcription
def split_audio_file(file_path, max_size_mb=20):
    # Implementation depends on audio processing library
    # Consider using pydub or similar for splitting
    pass
```

### Unsupported Format
```python
# Convert to supported format first
def convert_to_mp3(input_file):
    # Use ffmpeg or similar for conversion
    pass
```

### Rate Limiting
```python
import time

# Add delays between batch transcriptions
def batch_transcribe_with_delay(files, delay=1):
    results = []
    for file in files:
        result = audio_model.transcribe(file)
        results.append(result)
        time.sleep(delay)  # Prevent rate limiting
    return results
```

## Integration with Nerif Ecosystem

The AudioModel integrates seamlessly with other Nerif components:

```python
from nerif.model import AudioModel, SimpleChatModel
from nerif.core import nerif

# Transcribe and analyze
audio_model = AudioModel()
chat_model = SimpleChatModel()

transcription = audio_model.transcribe(Path("meeting.mp3"))

# Use nerif for quick analysis
if nerif(f"Does this transcription mention budget concerns? {transcription.text}"):
    # Generate detailed budget analysis
    budget_analysis = chat_model.chat(
        f"Extract all budget-related discussions from: {transcription.text}"
    )
    print(budget_analysis)
```

## Advanced Usage

### Custom Processing Pipeline

```python
from nerif.model import AudioModel
from nerif.batch import Batch, BatchFile
from pathlib import Path

# Create batch requests for multiple audio transcriptions
audio_files = list(Path("audio_folder").glob("*.mp3"))
requests = []

for i, audio_file in enumerate(audio_files):
    # Note: Batch API would need to support audio endpoints
    # This is a conceptual example
    requests.append({
        "custom_id": f"audio-{i}",
        "method": "POST",
        "url": "/v1/audio/transcriptions",
        "body": {
            "model": "whisper-1",
            "file": str(audio_file)
        }
    })

# Process in batch for efficiency
# (Actual implementation would depend on API support)
```

This AudioModel provides a simple yet powerful interface for audio transcription tasks, making it easy to convert speech to text for further processing and analysis within the Nerif ecosystem.