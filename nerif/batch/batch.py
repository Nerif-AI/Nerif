import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..model import SimpleChatModel, SimpleEmbeddingModel
from ..utils import LOGGER
from .file import BatchFile


class BatchStatus(str, Enum):
    """Batch job status enumeration matching OpenAI's API."""
    VALIDATING = "validating"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class Batch:
    """
    OpenAI-compatible Batch API implementation for Nerif.
    
    This class provides the same interface as OpenAI's Batch API:
    - Create batches from JSONL files
    - Track batch status
    - Retrieve results
    - List and cancel batches
    
    Example:
        # Create a batch
        batch = Batch.create(
            input_file_id="file-abc123",
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        # Check status
        status = Batch.retrieve(batch.id)
        
        # Get results when complete
        if status.status == "completed":
            output_file = batch.output_file_id
    """
    
    # Class-level storage for batch jobs (in production, use a database)
    _batches: Dict[str, Dict[str, Any]] = {}
    _storage_path = Path.home() / ".nerif" / "batches"
    _storage_path.mkdir(parents=True, exist_ok=True)
    
    def __init__(self):
        """Initialize Batch handler."""
        self.file_handler = BatchFile()
    
    @classmethod
    def create(
        cls,
        input_file_id: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new batch job.
        
        Args:
            input_file_id: The ID of the input file containing requests.
            endpoint: The API endpoint to use (e.g., "/v1/chat/completions").
            completion_window: Time window for completion (only "24h" supported).
            metadata: Optional metadata dictionary (up to 16 key-value pairs).
            
        Returns:
            Dictionary containing batch information:
                - id: Unique batch ID
                - object: "batch"
                - endpoint: API endpoint
                - errors: Error information (if any)
                - input_file_id: Input file ID
                - completion_window: Completion window
                - status: Current status
                - output_file_id: Output file ID (when complete)
                - error_file_id: Error file ID (if errors occur)
                - created_at: Creation timestamp
                - in_progress_at: Processing start timestamp
                - expires_at: Expiration timestamp
                - completed_at: Completion timestamp
                - failed_at: Failure timestamp (if failed)
                - expired_at: Expiration timestamp (if expired)
                - request_counts: Request statistics
                - metadata: User metadata
                
        Raises:
            ValueError: If invalid parameters are provided.
        """
        # Validate endpoint
        supported_endpoints = [
            "/v1/chat/completions",
            "/v1/embeddings",
            "/v1/completions"
        ]
        if endpoint not in supported_endpoints:
            raise ValueError(
                f"Unsupported endpoint '{endpoint}'. "
                f"Supported endpoints: {', '.join(supported_endpoints)}"
            )
        
        # Validate completion window
        if completion_window != "24h":
            raise ValueError("Only '24h' completion window is currently supported")
        
        # Validate metadata
        if metadata:
            if len(metadata) > 16:
                raise ValueError("Metadata can contain at most 16 key-value pairs")
            for key, value in metadata.items():
                if len(key) > 64:
                    raise ValueError(f"Metadata key '{key}' exceeds 64 character limit")
                if len(str(value)) > 512:
                    raise ValueError(f"Metadata value for '{key}' exceeds 512 character limit")
        
        # Create batch instance
        instance = cls()
        
        # Override file handler storage path if needed
        if hasattr(cls, '_storage_path'):
            instance.file_handler.storage_path = cls._storage_path
        
        # Validate input file exists
        try:
            requests = instance.file_handler.read_batch_file(input_file_id)
        except FileNotFoundError:
            raise ValueError(f"Input file '{input_file_id}' not found")
        
        # Generate batch ID
        batch_id = f"batch_{uuid.uuid4().hex[:24]}"
        
        # Calculate timestamps
        created_at = int(time.time())
        expires_at = created_at + (24 * 60 * 60)  # 24 hours
        
        # Create batch object
        batch_data = {
            "id": batch_id,
            "object": "batch",
            "endpoint": endpoint,
            "errors": None,
            "input_file_id": input_file_id,
            "completion_window": completion_window,
            "status": BatchStatus.VALIDATING,
            "output_file_id": None,
            "error_file_id": None,
            "created_at": created_at,
            "in_progress_at": None,
            "expires_at": expires_at,
            "finalizing_at": None,
            "completed_at": None,
            "failed_at": None,
            "expired_at": None,
            "cancelling_at": None,
            "cancelled_at": None,
            "request_counts": {
                "total": len(requests),
                "completed": 0,
                "failed": 0
            },
            "metadata": metadata or {}
        }
        
        # Store batch data
        cls._batches[batch_id] = batch_data
        instance._save_batch_state(batch_id, batch_data)
        
        # Start processing asynchronously
        asyncio.create_task(instance._process_batch(batch_id, requests, endpoint))
        
        return batch_data
    
    @classmethod
    def retrieve(cls, batch_id: str) -> Dict[str, Any]:
        """
        Retrieve information about a batch.
        
        Args:
            batch_id: The ID of the batch to retrieve.
            
        Returns:
            Dictionary containing current batch information.
            
        Raises:
            ValueError: If batch ID is not found.
        """
        if batch_id not in cls._batches:
            # Try to load from disk
            instance = cls()
            batch_data = instance._load_batch_state(batch_id)
            if batch_data:
                cls._batches[batch_id] = batch_data
            else:
                raise ValueError(f"Batch '{batch_id}' not found")
        
        return cls._batches[batch_id].copy()
    
    @classmethod
    def cancel(cls, batch_id: str) -> Dict[str, Any]:
        """
        Cancel a batch that is in progress.
        
        Args:
            batch_id: The ID of the batch to cancel.
            
        Returns:
            Updated batch information.
            
        Raises:
            ValueError: If batch cannot be cancelled.
        """
        batch = cls.retrieve(batch_id)
        
        # Check if batch can be cancelled
        if batch["status"] not in [BatchStatus.VALIDATING, BatchStatus.IN_PROGRESS]:
            raise ValueError(
                f"Batch '{batch_id}' cannot be cancelled. "
                f"Current status: {batch['status']}"
            )
        
        # Update status
        batch["status"] = BatchStatus.CANCELLING
        batch["cancelling_at"] = int(time.time())
        
        # Update stored data
        cls._batches[batch_id] = batch
        instance = cls()
        instance._save_batch_state(batch_id, batch)
        
        # Actually cancel the batch (in a real implementation, this would
        # stop the processing tasks)
        batch["status"] = BatchStatus.CANCELLED
        batch["cancelled_at"] = int(time.time())
        
        cls._batches[batch_id] = batch
        instance._save_batch_state(batch_id, batch)
        
        return batch
    
    @classmethod
    def list(
        cls,
        after: Optional[str] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        List batches with pagination.
        
        Args:
            after: Cursor for pagination (batch ID to start after).
            limit: Maximum number of batches to return (1-100).
            
        Returns:
            Dictionary containing:
                - data: List of batch objects
                - first_id: ID of first batch in list
                - last_id: ID of last batch in list
                - has_more: Whether more batches exist
                - object: "list"
        """
        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        
        # Load all batches from disk if not in memory
        instance = cls()
        instance._load_all_batches()
        
        # Sort batches by creation time (newest first)
        sorted_batches = sorted(
            cls._batches.values(),
            key=lambda b: b["created_at"],
            reverse=True
        )
        
        # Apply pagination
        if after:
            try:
                after_idx = next(
                    i for i, b in enumerate(sorted_batches)
                    if b["id"] == after
                )
                sorted_batches = sorted_batches[after_idx + 1:]
            except StopIteration:
                sorted_batches = []
        
        # Apply limit
        has_more = len(sorted_batches) > limit
        batch_list = sorted_batches[:limit]
        
        return {
            "object": "list",
            "data": batch_list,
            "first_id": batch_list[0]["id"] if batch_list else None,
            "last_id": batch_list[-1]["id"] if batch_list else None,
            "has_more": has_more
        }
    
    async def _process_batch(
        self,
        batch_id: str,
        requests: List[Dict[str, Any]],
        endpoint: str
    ) -> None:
        """
        Process a batch asynchronously.
        
        Args:
            batch_id: The batch ID being processed.
            requests: List of requests to process.
            endpoint: API endpoint to use.
        """
        batch = self._batches[batch_id]
        
        try:
            # Update status to in_progress
            batch["status"] = BatchStatus.IN_PROGRESS
            batch["in_progress_at"] = int(time.time())
            self._save_batch_state(batch_id, batch)
            
            # Process requests
            results = []
            errors = []
            
            # Determine which model to use based on endpoint
            if endpoint == "/v1/chat/completions":
                await self._process_chat_batch(batch_id, requests, results, errors)
            elif endpoint == "/v1/embeddings":
                await self._process_embedding_batch(batch_id, requests, results, errors)
            else:
                raise ValueError(f"Unsupported endpoint: {endpoint}")
            
            # Update to finalizing
            batch["status"] = BatchStatus.FINALIZING
            batch["finalizing_at"] = int(time.time())
            batch["request_counts"]["completed"] = len(results)
            batch["request_counts"]["failed"] = len(errors)
            self._save_batch_state(batch_id, batch)
            
            # Create output files
            if results:
                output_file = self.file_handler.create_output_file(batch_id, results)
                batch["output_file_id"] = output_file["id"]
            
            if errors:
                error_file = self.file_handler.create_error_file(batch_id, errors)
                batch["error_file_id"] = error_file["id"]
            
            # Update to completed
            batch["status"] = BatchStatus.COMPLETED
            batch["completed_at"] = int(time.time())
            self._save_batch_state(batch_id, batch)
            
        except Exception as e:
            # Update to failed
            batch["status"] = BatchStatus.FAILED
            batch["failed_at"] = int(time.time())
            batch["errors"] = {
                "object": "list",
                "data": [{
                    "code": "batch_processing_error",
                    "message": str(e)
                }]
            }
            self._save_batch_state(batch_id, batch)
            LOGGER.error(f"Batch {batch_id} failed: {e}")
    
    async def _process_chat_batch(
        self,
        batch_id: str,
        requests: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        errors: List[Dict[str, Any]]
    ) -> None:
        """Process chat completion requests in batch."""
        # Use thread pool for concurrent processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_request = {}
            for request in requests:
                future = executor.submit(self._process_single_chat, request)
                future_to_request[future] = request
            
            # Collect results
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    error = self._create_error_response(request, e)
                    errors.append(error)
                
                # Update progress
                batch = self._batches[batch_id]
                batch["request_counts"]["completed"] = len(results)
                batch["request_counts"]["failed"] = len(errors)
                
                # Check if cancelled
                if batch["status"] == BatchStatus.CANCELLING:
                    break
    
    async def _process_embedding_batch(
        self,
        batch_id: str,
        requests: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
        errors: List[Dict[str, Any]]
    ) -> None:
        """Process embedding requests in batch."""
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_request = {}
            for request in requests:
                future = executor.submit(self._process_single_embedding, request)
                future_to_request[future] = request
            
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    error = self._create_error_response(request, e)
                    errors.append(error)
                
                batch = self._batches[batch_id]
                batch["request_counts"]["completed"] = len(results)
                batch["request_counts"]["failed"] = len(errors)
                
                if batch["status"] == BatchStatus.CANCELLING:
                    break
    
    def _process_single_chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single chat completion request."""
        body = request["body"]
        
        # Extract parameters
        model_name = body.get("model", "gpt-4o")
        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.0)
        max_tokens = body.get("max_tokens")
        
        # Create model and get response
        model = SimpleChatModel(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Format messages for the model
        for msg in messages:
            model.messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Get response
        response = model.chat(messages[-1]["content"], append=False, max_tokens=max_tokens)
        
        # Create response in OpenAI format
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "custom_id": request["custom_id"],
            "response": {
                "status_code": 200,
                "request_id": f"req_{uuid.uuid4().hex[:24]}",
                "body": {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,  # Would need token counting
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
            },
            "error": None
        }
    
    def _process_single_embedding(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single embedding request."""
        body = request["body"]
        
        # Extract parameters
        model_name = body.get("model", "text-embedding-3-small")
        input_text = body.get("input", "")
        
        # Handle both string and list inputs
        if isinstance(input_text, list):
            input_text = " ".join(input_text)
        
        # Create model and get embedding
        model = SimpleEmbeddingModel(model=model_name)
        embedding = model.embed(input_text)
        
        # Create response in OpenAI format
        return {
            "id": f"embed-{uuid.uuid4().hex[:27]}",
            "custom_id": request["custom_id"],
            "response": {
                "status_code": 200,
                "request_id": f"req_{uuid.uuid4().hex[:24]}",
                "body": {
                    "object": "list",
                    "data": [{
                        "object": "embedding",
                        "index": 0,
                        "embedding": embedding
                    }],
                    "model": model_name,
                    "usage": {
                        "prompt_tokens": 0,
                        "total_tokens": 0
                    }
                }
            },
            "error": None
        }
    
    def _create_error_response(
        self,
        request: Dict[str, Any],
        exception: Exception
    ) -> Dict[str, Any]:
        """Create an error response for a failed request."""
        return {
            "custom_id": request["custom_id"],
            "response": {
                "status_code": 500,
                "request_id": f"req_{uuid.uuid4().hex[:24]}",
                "body": None
            },
            "error": {
                "message": str(exception),
                "type": type(exception).__name__,
                "code": "internal_error"
            }
        }
    
    def _save_batch_state(self, batch_id: str, batch_data: Dict[str, Any]) -> None:
        """Save batch state to disk."""
        state_file = self._storage_path / f"{batch_id}.json"
        with open(state_file, "w") as f:
            json.dump(batch_data, f, indent=2)
    
    def _load_batch_state(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Load batch state from disk."""
        state_file = self._storage_path / f"{batch_id}.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                return json.load(f)
        return None
    
    def _load_all_batches(self) -> None:
        """Load all batch states from disk."""
        for state_file in self._storage_path.glob("batch_*.json"):
            batch_id = state_file.stem
            if batch_id not in self._batches:
                batch_data = self._load_batch_state(batch_id)
                if batch_data:
                    self._batches[batch_id] = batch_data