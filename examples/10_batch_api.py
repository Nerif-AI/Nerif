#!/usr/bin/env python3
"""
Example 10: OpenAI-Compatible Batch API

This example demonstrates how to use Nerif's OpenAI-compatible Batch API
for processing large volumes of requests asynchronously at 50% lower cost.

The Batch API follows the same interface as OpenAI's Batch API:
https://platform.openai.com/docs/guides/batch
"""

import json
import time
from typing import List, Dict, Any

from nerif.batch import Batch, BatchFile


def create_chat_requests() -> List[Dict[str, Any]]:
    """Create sample chat completion requests."""
    return [
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "max_tokens": 100
            }
        },
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Explain quantum computing in simple terms"}
                ],
                "max_tokens": 200
            }
        },
        {
            "custom_id": "request-3",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Write a haiku about programming"}
                ],
                "max_tokens": 50
            }
        }
    ]


def create_embedding_requests() -> List[Dict[str, Any]]:
    """Create sample embedding requests."""
    return [
        {
            "custom_id": "embed-1",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": "text-embedding-3-small",
                "input": "Machine learning is a subset of artificial intelligence"
            }
        },
        {
            "custom_id": "embed-2",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": "text-embedding-3-small",
                "input": "Natural language processing helps computers understand human language"
            }
        }
    ]


def example_create_and_monitor_batch():
    """Example of creating a batch and monitoring its progress."""
    print("=== Creating and Monitoring Batch ===\n")
    
    # Step 1: Create a batch file with requests
    print("Step 1: Creating batch input file...")
    batch_file_handler = BatchFile()
    
    # Combine chat and embedding requests
    all_requests = create_chat_requests() + create_embedding_requests()
    
    # Create the input file
    input_file = batch_file_handler.create_batch_file(all_requests)
    print(f"Created input file: {input_file['id']}")
    print(f"File size: {input_file['bytes']} bytes")
    print(f"Total requests: {len(all_requests)}\n")
    
    # Step 2: Create a batch job
    print("Step 2: Creating batch job...")
    batch = Batch.create(
        input_file_id=input_file['id'],
        endpoint="/v1/chat/completions",  # Primary endpoint
        completion_window="24h",
        metadata={
            "description": "Example batch job",
            "created_by": "example_script"
        }
    )
    
    print(f"Created batch: {batch['id']}")
    print(f"Status: {batch['status']}")
    print(f"Total requests: {batch['request_counts']['total']}")
    print(f"Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(batch['created_at']))}")
    print(f"Expires at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(batch['expires_at']))}\n")
    
    # Step 3: Monitor batch progress
    print("Step 3: Monitoring batch progress...")
    
    # Poll for completion (in production, use webhooks or less frequent polling)
    max_wait_time = 60  # Maximum 60 seconds for demo
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        # Retrieve current batch status
        current_batch = Batch.retrieve(batch['id'])
        
        print(f"\rStatus: {current_batch['status']} | "
              f"Completed: {current_batch['request_counts']['completed']} | "
              f"Failed: {current_batch['request_counts']['failed']}",
              end='', flush=True)
        
        if current_batch['status'] in ['completed', 'failed', 'cancelled']:
            print("\n")
            break
        
        time.sleep(2)  # Poll every 2 seconds
    
    # Step 4: Retrieve results
    print("\nStep 4: Retrieving results...")
    final_batch = Batch.retrieve(batch['id'])
    
    print(f"Final status: {final_batch['status']}")
    print(f"Completed requests: {final_batch['request_counts']['completed']}")
    print(f"Failed requests: {final_batch['request_counts']['failed']}")
    
    if final_batch['output_file_id']:
        print(f"\nOutput file ID: {final_batch['output_file_id']}")
        
        # Read and display results
        print("\nSample results:")
        output_file_path = batch_file_handler.storage_path / f"{final_batch['output_file_id']}_output.jsonl"
        
        if output_file_path.exists():
            with open(output_file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 3:  # Show first 3 results
                        break
                    result = json.loads(line)
                    print(f"\nRequest ID: {result['custom_id']}")
                    if result['response']['status_code'] == 200:
                        if 'choices' in result['response']['body']:
                            # Chat completion result
                            content = result['response']['body']['choices'][0]['message']['content']
                            print(f"Response: {content[:100]}...")
                        elif 'data' in result['response']['body']:
                            # Embedding result
                            embedding = result['response']['body']['data'][0]['embedding']
                            print(f"Embedding (first 5 values): {embedding[:5]}")
    
    if final_batch['error_file_id']:
        print(f"\nError file ID: {final_batch['error_file_id']}")


def example_list_batches():
    """Example of listing all batches."""
    print("\n\n=== Listing Batches ===\n")
    
    # List all batches
    batch_list = Batch.list(limit=5)
    
    print(f"Total batches shown: {len(batch_list['data'])}")
    print(f"Has more: {batch_list['has_more']}\n")
    
    for batch in batch_list['data']:
        print(f"Batch ID: {batch['id']}")
        print(f"  Status: {batch['status']}")
        print(f"  Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(batch['created_at']))}")
        print(f"  Requests: {batch['request_counts']['total']} total, "
              f"{batch['request_counts']['completed']} completed, "
              f"{batch['request_counts']['failed']} failed")
        if batch['metadata']:
            print(f"  Metadata: {batch['metadata']}")
        print()


def example_cancel_batch():
    """Example of cancelling a batch."""
    print("\n\n=== Cancelling a Batch ===\n")
    
    # Create a new batch to cancel
    batch_file_handler = BatchFile()
    requests = create_chat_requests()
    input_file = batch_file_handler.create_batch_file(requests)
    
    batch = Batch.create(
        input_file_id=input_file['id'],
        endpoint="/v1/chat/completions",
        metadata={"purpose": "cancellation_demo"}
    )
    
    print(f"Created batch: {batch['id']}")
    print(f"Initial status: {batch['status']}")
    
    # Cancel the batch
    print("\nCancelling batch...")
    cancelled_batch = Batch.cancel(batch['id'])
    
    print(f"Final status: {cancelled_batch['status']}")
    print(f"Cancelled at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cancelled_batch['cancelled_at']))}")


def example_batch_with_mixed_endpoints():
    """Example of processing requests for different endpoints."""
    print("\n\n=== Batch with Mixed Endpoints ===\n")
    
    # Create requests for different endpoints
    mixed_requests = [
        {
            "custom_id": "chat-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 50
            }
        },
        {
            "custom_id": "embed-1",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": "text-embedding-3-small",
                "input": "Sample text for embedding"
            }
        },
        {
            "custom_id": "chat-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "How are you?"}],
                "max_tokens": 50
            }
        }
    ]
    
    # Create batch file and job
    batch_file_handler = BatchFile()
    input_file = batch_file_handler.create_batch_file(mixed_requests)
    
    batch = Batch.create(
        input_file_id=input_file['id'],
        endpoint="/v1/chat/completions",  # Primary endpoint
        completion_window="24h"
    )
    
    print(f"Created batch with mixed endpoints: {batch['id']}")
    print(f"Total requests: {batch['request_counts']['total']}")
    
    # Wait for completion
    print("\nProcessing...")
    time.sleep(5)  # Give it some time to process
    
    # Check results
    final_batch = Batch.retrieve(batch['id'])
    print(f"Status: {final_batch['status']}")
    print(f"Completed: {final_batch['request_counts']['completed']}")
    print(f"Failed: {final_batch['request_counts']['failed']}")


def main():
    """Run all batch API examples."""
    print("Nerif Batch API Examples")
    print("========================\n")
    print("This demonstrates OpenAI-compatible batch processing.\n")
    
    # Run examples
    example_create_and_monitor_batch()
    example_list_batches()
    example_cancel_batch()
    example_batch_with_mixed_endpoints()
    
    print("\n\nAll examples completed!")
    print("\nNote: In production, batches process asynchronously and may take")
    print("up to 24 hours. Use webhooks or periodic polling to check status.")


if __name__ == "__main__":
    main()