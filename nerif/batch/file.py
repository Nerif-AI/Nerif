import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BatchFile:
    """
    Handles batch file operations for OpenAI-compatible batch API.
    Manages JSONL file creation, validation, and parsing.
    """
    
    SUPPORTED_ENDPOINTS = [
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/completions",
    ]
    
    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize BatchFile handler.
        
        Args:
            file_path: Path to store batch files. If None, uses temp directory.
        """
        if file_path:
            self.storage_path = Path(file_path)
        else:
            self.storage_path = Path.home() / ".nerif" / "batches"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def create_batch_file(
        self,
        requests: List[Dict[str, Any]],
        file_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a JSONL batch file from a list of requests.
        
        Args:
            requests: List of request dictionaries, each containing:
                - custom_id: A unique identifier for the request
                - method: HTTP method (POST)
                - url: API endpoint URL
                - body: Request body with model parameters
            file_id: Optional file ID. If not provided, generates a UUID.
            
        Returns:
            Dictionary with file metadata including:
                - id: File ID
                - object: "file"
                - bytes: File size in bytes
                - created_at: Unix timestamp
                - filename: Name of the file
                - purpose: "batch"
                - status: "processed"
                
        Example request format:
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 100
                }
            }
        """
        # Validate requests
        for i, request in enumerate(requests):
            self._validate_request(request, i)
        
        # Generate file ID if not provided
        if file_id is None:
            file_id = f"file-{uuid.uuid4().hex[:24]}"
        
        # Create JSONL file
        filename = f"{file_id}.jsonl"
        file_path = self.storage_path / filename
        
        with open(file_path, "w") as f:
            for request in requests:
                json.dump(request, f)
                f.write("\n")
        
        # Get file stats
        file_stats = file_path.stat()
        
        return {
            "id": file_id,
            "object": "file",
            "bytes": file_stats.st_size,
            "created_at": int(file_stats.st_ctime),
            "filename": filename,
            "purpose": "batch",
            "status": "processed",
            "status_details": None
        }
    
    def read_batch_file(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Read and parse a batch JSONL file.
        
        Args:
            file_id: The file ID to read.
            
        Returns:
            List of request dictionaries.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        filename = f"{file_id}.jsonl"
        file_path = self.storage_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Batch file {file_id} not found")
        
        requests = []
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    request = json.loads(line)
                    requests.append(request)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON on line {line_num}: {e.msg}",
                        e.doc,
                        e.pos
                    )
        
        return requests
    
    def create_output_file(
        self,
        batch_id: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create an output file with batch results.
        
        Args:
            batch_id: The batch ID this output belongs to.
            results: List of result dictionaries.
            
        Returns:
            Dictionary with output file metadata.
        """
        file_id = f"file-{uuid.uuid4().hex[:24]}"
        filename = f"{file_id}_output.jsonl"
        file_path = self.storage_path / filename
        
        with open(file_path, "w") as f:
            for result in results:
                json.dump(result, f)
                f.write("\n")
        
        file_stats = file_path.stat()
        
        return {
            "id": file_id,
            "object": "file",
            "bytes": file_stats.st_size,
            "created_at": int(file_stats.st_ctime),
            "filename": filename,
            "purpose": "batch_output",
            "status": "processed",
            "batch_id": batch_id
        }
    
    def create_error_file(
        self,
        batch_id: str,
        errors: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Create an error file for failed requests.
        
        Args:
            batch_id: The batch ID this error file belongs to.
            errors: List of error dictionaries.
            
        Returns:
            Dictionary with error file metadata, or None if no errors.
        """
        if not errors:
            return None
        
        file_id = f"file-{uuid.uuid4().hex[:24]}"
        filename = f"{file_id}_errors.jsonl"
        file_path = self.storage_path / filename
        
        with open(file_path, "w") as f:
            for error in errors:
                json.dump(error, f)
                f.write("\n")
        
        file_stats = file_path.stat()
        
        return {
            "id": file_id,
            "object": "file",
            "bytes": file_stats.st_size,
            "created_at": int(file_stats.st_ctime),
            "filename": filename,
            "purpose": "batch_error",
            "status": "processed",
            "batch_id": batch_id
        }
    
    def delete_file(self, file_id: str) -> bool:
        """
        Delete a batch file.
        
        Args:
            file_id: The file ID to delete.
            
        Returns:
            True if file was deleted, False if not found.
        """
        # Try different file extensions
        for ext in [".jsonl", "_output.jsonl", "_errors.jsonl"]:
            filename = f"{file_id}{ext}"
            file_path = self.storage_path / filename
            if file_path.exists():
                file_path.unlink()
                return True
        
        return False
    
    def _validate_request(self, request: Dict[str, Any], index: int) -> None:
        """
        Validate a single batch request.
        
        Args:
            request: Request dictionary to validate.
            index: Request index for error messages.
            
        Raises:
            ValueError: If request is invalid.
        """
        required_fields = ["custom_id", "method", "url", "body"]
        for field in required_fields:
            if field not in request:
                raise ValueError(
                    f"Request {index}: Missing required field '{field}'"
                )
        
        # Validate method
        if request["method"] != "POST":
            raise ValueError(
                f"Request {index}: Only POST method is supported"
            )
        
        # Validate URL
        url = request["url"]
        if not any(url.endswith(endpoint) for endpoint in self.SUPPORTED_ENDPOINTS):
            raise ValueError(
                f"Request {index}: Unsupported endpoint '{url}'. "
                f"Supported endpoints: {', '.join(self.SUPPORTED_ENDPOINTS)}"
            )
        
        # Validate body
        body = request["body"]
        if not isinstance(body, dict):
            raise ValueError(
                f"Request {index}: Body must be a dictionary"
            )
        
        # Validate model is present
        if "model" not in body:
            raise ValueError(
                f"Request {index}: Missing 'model' in request body"
            )
        
        # Validate custom_id length
        custom_id = request["custom_id"]
        if len(custom_id) > 64:
            raise ValueError(
                f"Request {index}: custom_id must be 64 characters or less"
            )