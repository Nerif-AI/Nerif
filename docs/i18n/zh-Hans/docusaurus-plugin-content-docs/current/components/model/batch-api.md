# Nerif Batch API（OpenAI 兼容）

Nerif 提供了与 OpenAI 兼容的 Batch API，允许你异步处理大量请求。此实现遵循与 [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) 相同的接口。

## 概述

Batch API 支持以下功能：
- 在单个批处理文件中发送多个请求
- 以低 50% 的成本异步处理请求
- 在 24 小时完成窗口内处理大规模操作
- 跟踪批处理状态并获取结果

## 安装

Batch API 包含在 Nerif 包中：

```bash
pip install nerif
```

## 快速开始

### 1. 创建批处理输入文件

首先，创建一个包含请求的 JSONL 文件：

```python
from nerif.batch import BatchFile

# Define your requests
requests = [
    {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "max_tokens": 100
        }
    },
    {
        "custom_id": "request-2",
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {
            "model": "text-embedding-3-small",
            "input": "Hello world"
        }
    }
]

# Create the batch file
batch_file = BatchFile()
input_file = batch_file.create_batch_file(requests)
print(f"Created input file: {input_file['id']}")
```

### 2. 创建批处理任务

```python
from nerif.batch import Batch

# Create a batch job
batch = Batch.create(
    input_file_id=input_file['id'],
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "My batch job"
    }
)

print(f"Created batch: {batch['id']}")
print(f"Status: {batch['status']}")
```

### 3. 监控批处理进度

```python
import time

# Check batch status
while True:
    batch_status = Batch.retrieve(batch['id'])
    print(f"Status: {batch_status['status']}")

    if batch_status['status'] in ['completed', 'failed', 'cancelled']:
        break

    time.sleep(10)  # Check every 10 seconds

# Get results
if batch_status['status'] == 'completed':
    print(f"Output file: {batch_status['output_file_id']}")
    print(f"Completed: {batch_status['request_counts']['completed']}")
    print(f"Failed: {batch_status['request_counts']['failed']}")
```

## API 参考

### BatchFile 类

处理批处理请求的 JSONL 文件操作。

```python
# Create a batch file
file_info = batch_file.create_batch_file(requests, file_id=None)

# Read a batch file
requests = batch_file.read_batch_file(file_id)

# Create output/error files
output_file = batch_file.create_output_file(batch_id, results)
error_file = batch_file.create_error_file(batch_id, errors)
```

### Batch 类

批处理操作的主接口，完全匹配 OpenAI 的 API。

#### 创建批处理

```python
batch = Batch.create(
    input_file_id="file-abc123",
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"key": "value"}  # Optional, max 16 pairs
)
```

#### 查询批处理

```python
batch = Batch.retrieve("batch_abc123")
```

#### 取消批处理

```python
cancelled_batch = Batch.cancel("batch_abc123")
```

#### 列出批处理

```python
batches = Batch.list(
    after="batch_abc123",  # Optional pagination cursor
    limit=20  # 1-100
)
```

## 请求格式

JSONL 文件中的每个请求必须遵循以下格式：

```json
{
    "custom_id": "unique-request-id",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Your message here"}
        ],
        "max_tokens": 100
    }
}
```

### 支持的端点

- `/v1/chat/completions` - 聊天补全
- `/v1/embeddings` - 文本嵌入
- `/v1/completions` - 文本补全（旧版）

## 批处理对象

批处理对象包含以下内容：

```python
{
    "id": "batch_abc123",
    "object": "batch",
    "endpoint": "/v1/chat/completions",
    "errors": null,
    "input_file_id": "file-abc123",
    "completion_window": "24h",
    "status": "completed",
    "output_file_id": "file-def456",
    "error_file_id": null,
    "created_at": 1714508499,
    "in_progress_at": 1714508500,
    "expires_at": 1714594899,
    "completed_at": 1714508510,
    "request_counts": {
        "total": 100,
        "completed": 98,
        "failed": 2
    },
    "metadata": {
        "description": "Nightly job"
    }
}
```

### 批处理状态值

- `validating` - 正在验证输入文件
- `failed` - 输入验证失败
- `in_progress` - 正在处理请求
- `finalizing` - 正在生成结果文件
- `completed` - 批处理已成功完成
- `expired` - 批处理已过期（24 小时窗口）
- `cancelling` - 正在取消批处理
- `cancelled` - 批处理已被取消

## 输出格式

结果以 JSONL 格式保存：

```json
{
    "id": "response-1",
    "custom_id": "request-1",
    "response": {
        "status_code": 200,
        "request_id": "req_123",
        "body": {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1714508505,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Paris is the capital of France."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        }
    },
    "error": null
}
```

## 错误处理

失败的请求保存在单独的错误文件中：

```json
{
    "custom_id": "request-2",
    "response": {
        "status_code": 500,
        "request_id": "req_124",
        "body": null
    },
    "error": {
        "message": "Internal server error",
        "type": "server_error",
        "code": "internal_error"
    }
}
```

## 最佳实践

1. **批处理大小**：虽然没有硬性限制，但建议保持合理的批处理规模（例如 1000-5000 个请求）
2. **完成窗口**：所有批处理都有 24 小时的完成窗口
3. **元数据**：使用元数据跟踪批处理目的、来源等（最多 16 个键值对）
4. **错误处理**：始终检查输出文件和错误文件
5. **轮询**：在生产环境中，降低轮询频率（例如每几分钟一次）或使用 webhook

## 示例：处理多个文件

```python
from nerif.batch import Batch, BatchFile
import json

# Process multiple text files
texts = ["file1.txt", "file2.txt", "file3.txt"]
requests = []

for i, filename in enumerate(texts):
    with open(filename, 'r') as f:
        content = f.read()

    requests.append({
        "custom_id": f"summarize-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Summarize the following text"},
                {"role": "user", "content": content}
            ],
            "max_tokens": 200
        }
    })

# Create and process batch
batch_file = BatchFile()
input_file = batch_file.create_batch_file(requests)
batch = Batch.create(
    input_file_id=input_file['id'],
    endpoint="/v1/chat/completions"
)

print(f"Processing {len(requests)} files in batch {batch['id']}")
```

## 限制

- 批处理必须在 24 小时内完成
- 仅支持 POST 请求
- 仅限于特定端点（聊天、嵌入、补全）
- 结果异步处理（非实时）

## 存储

默认情况下，批处理文件存储在 `~/.nerif/batches/`。你可以自定义路径：

```python
batch_file = BatchFile(file_path="/custom/path/to/batches")
```
