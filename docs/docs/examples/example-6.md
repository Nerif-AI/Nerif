# Example 6: Batch API

```python
from nerif.batch import Batch, BatchFile
import time

# Create batch requests
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

# Create batch file
batch_file = BatchFile()
input_file = batch_file.create_batch_file(requests)
print(f"Created input file: {input_file['id']}")

# Create batch job
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

# Monitor batch progress
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