# Nerif Batch API (OpenAI-Compatible)

Nerif provides an OpenAI-compatible Batch API that allows you to process large volumes of requests asynchronously. This implementation follows the same interface as [OpenAI's Batch API](https://platform.openai.com/docs/guides/batch).

## Overview

The Batch API enables you to:
- Send multiple requests in a single batch file
- Process requests asynchronously at 50% lower cost
- Handle large-scale operations with a 24-hour completion window
- Track batch status and retrieve results

## Installation

The Batch API is included in the Nerif package:

```bash
pip install nerif
```

## Quick Start

### 1. Create a Batch Input File

First, create a JSONL file with your requests:

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

### 2. Create a Batch Job

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

### 3. Monitor Batch Progress

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

## API Reference

### BatchFile Class

Handles JSONL file operations for batch requests.

```python
# Create a batch file
file_info = batch_file.create_batch_file(requests, file_id=None)

# Read a batch file
requests = batch_file.read_batch_file(file_id)

# Create output/error files
output_file = batch_file.create_output_file(batch_id, results)
error_file = batch_file.create_error_file(batch_id, errors)
```

### Batch Class

Main interface for batch operations, matching OpenAI's API exactly.

#### Create a Batch

```python
batch = Batch.create(
    input_file_id="file-abc123",
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"key": "value"}  # Optional, max 16 pairs
)
```

#### Retrieve a Batch

```python
batch = Batch.retrieve("batch_abc123")
```

#### Cancel a Batch

```python
cancelled_batch = Batch.cancel("batch_abc123")
```

#### List Batches

```python
batches = Batch.list(
    after="batch_abc123",  # Optional pagination cursor
    limit=20  # 1-100
)
```

## Request Format

Each request in the JSONL file must follow this format:

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

### Supported Endpoints

- `/v1/chat/completions` - Chat completions
- `/v1/embeddings` - Text embeddings
- `/v1/completions` - Text completions (legacy)

## Batch Object

A batch object contains:

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

### Batch Status Values

- `validating` - Validating input file
- `failed` - Input validation failed
- `in_progress` - Processing requests
- `finalizing` - Generating result files
- `completed` - Batch completed successfully
- `expired` - Batch expired (24h window)
- `cancelling` - Batch is being cancelled
- `cancelled` - Batch was cancelled

## Output Format

Results are saved in JSONL format:

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

## Error Handling

Failed requests are saved in a separate error file:

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

## Best Practices

1. **Batch Size**: While there's no hard limit, keep batches reasonable (e.g., 1000-5000 requests)
2. **Completion Window**: All batches have a 24-hour completion window
3. **Metadata**: Use metadata to track batch purpose, source, etc. (max 16 key-value pairs)
4. **Error Handling**: Always check both output and error files
5. **Polling**: For production, poll less frequently (e.g., every few minutes) or use webhooks

## Example: Processing Multiple Files

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

## Limitations

- Batches must complete within 24 hours
- Only POST requests are supported
- Limited to specific endpoints (chat, embeddings, completions)
- Results are processed asynchronously (not real-time)

## Storage

By default, batch files are stored in `~/.nerif/batches/`. You can customize this:

```python
batch_file = BatchFile(file_path="/custom/path/to/batches")
```