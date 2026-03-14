---
sidebar_position: 1
---

# Nerif 模型

## 模型类

### SimpleChatModel

Nerif 中的主要模型类。支持文本和多模态输入（图片、音频、视频），以及工具调用和结构化输出（JSON 模式）。

属性：

- `model (str)`：使用的模型名称（默认值：`NERIF_DEFAULT_LLM_MODEL`）。
- `default_prompt (str)`：对话的默认系统提示词。
- `temperature (float)`：响应生成的温度设置。
- `counter (NerifTokenCounter)`：Token 计数器实例。
- `messages (List[Any])`：对话历史记录。
- `max_tokens (int)`：响应中生成的最大 token 数。
- `retry_config (RetryConfig)`：API 调用的重试策略。

方法：

- `reset(prompt=None)`：重置对话历史。可选设置新的系统提示词。
- `set_max_tokens(max_tokens=None|int)`：设置最大 token 限制。
- `chat(message, append=False, max_tokens=None, tools=None, tool_choice=None, response_format=None)`：发送消息并获取响应。
- `stream_chat(message, append=False, max_tokens=None, response_format=None)`：流式返回响应 token。
- `achat(...)`：chat() 的异步版本。
- `astream_chat(...)`：异步流式输出。

初始化：

```python
def __init__(
    self,
    model: str = NERIF_DEFAULT_LLM_MODEL,
    default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
    temperature: float = 0.0,
    counter: NerifTokenCounter = None,
    max_tokens: None | int = None,
    retry_config: RetryConfig = None,
)
```

#### `chat()` 方法

```python
def chat(
    self,
    message: Union[str, MultiModalMessage],
    append: bool = False,
    max_tokens: None | int = None,
    tools: Optional[List[Union[Dict, ToolDefinition]]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[Any] = None,
    response_model: Optional[type] = None,
) -> Union[str, List[ToolCallResult]]:
```

**参数：**
- `message`：文本字符串或 `MultiModalMessage`（多模态输入）。
- `append`：如果为 `True`，保留对话历史；如果为 `False`，在响应后重置。
- `max_tokens`：覆盖本次请求的最大 token 数。
- `tools`：工具定义列表，用于函数调用（参见[工具调用](#工具调用)）。
- `tool_choice`：工具选择参数（例如 `"auto"`、`"none"` 或指定工具）。
- `response_format`：响应格式（例如 `{"type": "json_object"}` 用于 JSON 模式）。

**返回值：** 文本响应字符串，或在调用工具时返回 `List[ToolCallResult]`。

示例：

```python
from nerif.model import SimpleChatModel

model = SimpleChatModel()

print(model.chat("What is the capital of the moon?"))
print(model.chat("What is the capital of the moon?", max_tokens=10))
```

---

### MultiModalMessage

用于构建包含文本、图片、音频和视频的多模态消息的辅助类。采用流式（链式）API。

方法：

- `add_text(text: str)`：添加文本内容。
- `add_image_url(url: str)`：通过 URL 添加图片。
- `add_image_path(path: str)`：通过本地文件路径添加图片（自动进行 base64 编码）。
- `add_image_base64(b64: str, media_type="image/jpeg")`：添加 base64 编码的图片。
- `add_audio_url(url: str)`：通过 URL 添加音频。
- `add_audio_path(path: str, format="wav")`：通过本地文件添加音频（自动进行 base64 编码）。
- `add_audio_base64(b64: str, format="wav")`：添加 base64 编码的音频。
- `add_video_url(url: str)`：通过 URL 添加视频。
- `add_video_path(path: str)`：通过本地文件添加视频（自动进行 base64 编码）。
- `to_content()`：返回用于 API 的内容部分列表。

所有 `add_*` 方法都返回 `self`，因此可以链式调用：

```python
from nerif.model import SimpleChatModel, MultiModalMessage

model = SimpleChatModel(model="gpt-4o")

msg = (
    MultiModalMessage()
    .add_text("What do you see in this image?")
    .add_image_url("https://example.com/photo.jpg")
)

result = model.chat(msg)
print(result)
```

---

### 工具调用

Nerif 通过 `ToolDefinition` 和 `ToolCallResult` 支持 OpenAI 兼容的工具调用。

#### ToolDefinition

用于以 OpenAI 函数调用格式定义工具的辅助类。

```python
from nerif.model import SimpleChatModel, ToolDefinition

weather_tool = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        "required": ["location"],
    },
)

model = SimpleChatModel(model="gpt-4o")
result = model.chat(
    "What's the weather like in San Francisco?",
    tools=[weather_tool],
    tool_choice="auto",
)
```

#### ToolCallResult

表示模型返回的工具调用。当模型决定调用工具而非生成文本时返回此对象。

属性：
- `id (str)`：此工具调用的唯一 ID。
- `name (str)`：要调用的函数名称。
- `arguments (str)`：函数参数的 JSON 字符串。

```python
if isinstance(result, list):
    for tc in result:
        print(f"Tool: {tc.name}, Args: {tc.arguments}")
```

---

### 结构化输出（JSON 模式）

使用 `response_format` 从模型获取结构化 JSON 输出：

```python
from nerif.model import SimpleChatModel
from nerif.utils import NerifFormat

model = SimpleChatModel(model="gpt-4o")
result = model.chat(
    "List three programming languages with their year of creation. "
    "Respond in JSON format as an array of objects with 'name' and 'year' fields.",
    response_format={"type": "json_object"},
)

# Parse robustly with NerifFormat
parsed = NerifFormat.json_parse(result)
```

---

### 流式输出

使用 `stream_chat()` 实现实时逐 token 输出：

```python
from nerif.model import SimpleChatModel

model = SimpleChatModel()

for chunk in model.stream_chat("写一首关于编程的俳句。"):
    print(chunk, end="", flush=True)
print()
```

### 异步支持

所有模型方法都有以 `a` 为前缀的异步对应版本：

```python
import asyncio
from nerif.model import SimpleChatModel

async def main():
    model = SimpleChatModel()
    result = await model.achat("你好！")
    print(result)

    # 异步流式输出
    async for chunk in model.astream_chat("给我讲个故事。"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### 重试配置

配置指数退避的自动重试：

```python
from nerif.model import SimpleChatModel
from nerif.utils import RetryConfig, NO_RETRY, AGGRESSIVE_RETRY

# 自定义重试
model = SimpleChatModel(retry_config=RetryConfig(max_retries=5, base_delay=0.5))

# 不重试
model_fast = SimpleChatModel(retry_config=NO_RETRY)
```

### Pydantic 结构化输出

使用 `response_model` 获取类型安全的结构化输出：

```python
from pydantic import BaseModel
from nerif.model import SimpleChatModel

class City(BaseModel):
    name: str
    country: str
    population: int

model = SimpleChatModel()
city = model.chat("介绍一下东京。", response_model=City)
print(f"{city.name}，{city.country}：{city.population:,}")
```

---

### SimpleEmbeddingModel

用于文本嵌入的简单模型类。将文本字符串转换为数值向量表示。

属性：

- `model (str)`：使用的嵌入模型名称。
- `counter (NerifTokenCounter)`：可选的 token 使用量计数器。

方法：

- `embed(string: str) -> List[float]`：将字符串编码为嵌入向量。
- `aembed(string: str) -> List[float]`：embed() 的异步版本。

初始化：
```python
def __init__(
    self,
    model: str = "text-embedding-3-small",
    counter: Optional[NerifTokenCounter] = None,
)
```

示例：

```python
from nerif.model import SimpleEmbeddingModel

embedding_model = SimpleEmbeddingModel()
print(embedding_model.embed("What is the capital of the moon?"))
```

---

### LogitsChatModel

用于从 LLM 获取 logits（对数概率）的模型。Nerif Core 内部使用此模型进行基于 logits 的判断。

属性：

- `model (str)`：使用的模型名称。
- `default_prompt (str)`：对话的默认系统提示词。
- `temperature (float)`：响应生成的温度设置。
- `counter (NerifTokenCounter)`：Token 计数器实例。
- `max_tokens (int)`：生成的最大 token 数。

方法：

- `reset()`：重置对话历史。
- `set_max_tokens(max_tokens: None|int)`：设置最大 token 限制。
- `chat(message: str, max_tokens: None|int, logprobs: bool = True, top_logprobs: int = 5) -> Any`：
    发送消息并获取带有 logit 概率的响应。

初始化：

```python
def __init__(
    self,
    model: str = NERIF_DEFAULT_LLM_MODEL,
    default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
    temperature: float = 0.0,
    counter: Optional[NerifTokenCounter] = None,
    max_tokens: int | None = None,
)
```

示例：

```python
from nerif.model import LogitsChatModel

logits_model = LogitsChatModel()
print(logits_model.chat("What is the capital of the moon?"))
```

---

### OllamaEmbeddingModel

用于 Ollama 本地模型的嵌入模型。

属性：

- `model (str)`：Ollama 模型名称（默认值：`"ollama/mxbai-embed-large"`）。
- `url (str)`：Ollama API 地址（默认值：`"http://localhost:11434/v1/"`）。
- `counter (NerifTokenCounter)`：可选的 token 使用量计数器。

方法：

- `embed(string: str) -> List[float]`：将字符串编码为嵌入向量。

```python
from nerif.model import OllamaEmbeddingModel

model = OllamaEmbeddingModel(model="ollama/mxbai-embed-large")
embedding = model.embed("Hello world")
```

---

### VideoModel

用于视频理解任务的模型。封装了 `SimpleChatModel` 和 `MultiModalMessage` 以支持视频输入。

方法：

- `analyze_url(video_url: str, prompt: str = "Describe this video.", max_tokens: int | None = None) -> str`：通过 URL 分析视频。
- `analyze_path(video_path: str, prompt: str = "Describe this video.", max_tokens: int | None = None) -> str`：通过本地文件分析视频。
- `reset()`：重置对话历史。

```python
from nerif.model import VideoModel

video_model = VideoModel(model="gpt-4o")
result = video_model.analyze_url("https://example.com/video.mp4", "What happens in this video?")
print(result)
```

---

### VisionModel

用于图像分析的视觉模型。对于新代码，建议使用 `SimpleChatModel` 配合 `MultiModalMessage`。

详细文档请参见[视觉模型](./vision-model.md)。
