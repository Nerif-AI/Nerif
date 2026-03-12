# Nerif 音频模型

Nerif 提供了简洁而强大的 AudioModel 类，用于音频处理任务，尤其是使用 OpenAI Whisper 模型进行音频转录。

## 概述

AudioModel 类支持以下功能：
- 使用先进的语音识别技术将音频文件转录为文本
- 支持多种音频格式（mp3、mp4、mpeg、mpga、m4a、wav、webm）
- 处理最大 25MB 的音频文件

## 安装

AudioModel 包含在 Nerif 基础包中：

```bash
pip install nerif
```

## 快速开始

```python
from nerif.model import AudioModel
from pathlib import Path

# Initialize the audio model
audio_model = AudioModel()

# Transcribe an audio file
transcription = audio_model.transcribe(Path("interview.mp3"))
print(transcription.text)
```

## API 参考

### AudioModel 类

用于音频任务的简单代理，包括转录和语音转文本。

#### 初始化

```python
audio_model = AudioModel()
```

AudioModel 会自动使用你的环境凭证初始化 OpenAI 客户端。

#### 方法

##### transcribe(file: Path)

使用 Whisper-1 模型将音频文件转录为文本。

**参数：**
- `file` (Path)：要转录的音频文件路径

**返回值：**
- 转录对象，包含：
  - `text`：转录的文本内容
  - 转录的附加元数据

**示例：**
```python
from pathlib import Path

# Transcribe a single file
result = audio_model.transcribe(Path("podcast.mp3"))
print(f"Transcription: {result.text}")
```

## 支持的音频格式

AudioModel 支持以下音频格式：
- mp3
- mp4
- mpeg
- mpga
- m4a
- wav
- webm

## 文件大小限制

- 最大文件大小：25MB
- 对于较大的文件，建议将其分割为较小的片段

## 示例

### 基础转录

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

### 批量转录

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

### 带错误处理的转录

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

### 与文本处理集成

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

### 创建字幕

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

## 最佳实践

1. **文件格式**：使用 WAV 或 MP3 以获得最佳兼容性
2. **音频质量**：确保音频清晰，背景噪音最小
3. **文件大小**：保持文件在 25MB 以下；较长的录音请分割后处理
4. **错误处理**：文件操作始终应实现错误处理
5. **批量处理**：按顺序处理多个文件以避免速率限制

## 常见问题与解决方案

### 文件过大
```python
# Split large audio files before transcription
def split_audio_file(file_path, max_size_mb=20):
    # Implementation depends on audio processing library
    # Consider using pydub or similar for splitting
    pass
```

### 不支持的格式
```python
# Convert to supported format first
def convert_to_mp3(input_file):
    # Use ffmpeg or similar for conversion
    pass
```

### 速率限制
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

## 与 Nerif 生态系统集成

AudioModel 与其他 Nerif 组件无缝集成：

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

## 高级用法

### 自定义处理流水线

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

AudioModel 为音频转录任务提供了简洁而强大的接口，可以轻松地将语音转换为文本，以便在 Nerif 生态系统中进行进一步处理和分析。
