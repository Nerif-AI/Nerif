import unittest
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from nerif.batch import Batch, BatchFile
from nerif.batch.batch import BatchStatus


class TestBatchFile(unittest.TestCase):
    """Test cases for BatchFile operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.batch_file = BatchFile(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_valid_batch_file(self):
        """Test creating a valid batch file."""
        requests = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "user", "content": "Hello, world!"}
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
                    "input": "Sample text for embedding"
                }
            }
        ]
        
        file_info = self.batch_file.create_batch_file(requests)
        
        # Verify file info
        self.assertEqual(file_info["object"], "file")
        self.assertEqual(file_info["purpose"], "batch")
        self.assertEqual(file_info["status"], "processed")
        self.assertIn("file-", file_info["id"])
        self.assertGreater(file_info["bytes"], 0)
        
        # Verify file contents
        read_requests = self.batch_file.read_batch_file(file_info["id"])
        self.assertEqual(len(read_requests), 2)
        self.assertEqual(read_requests[0]["custom_id"], "request-1")
        self.assertEqual(read_requests[1]["custom_id"], "request-2")
    
    def test_create_batch_file_with_invalid_request(self):
        """Test creating batch file with invalid requests."""
        # Missing required field
        requests = [{
            "custom_id": "request-1",
            "method": "POST",
            # Missing 'url' field
            "body": {"model": "gpt-4o"}
        }]
        
        with self.assertRaises(ValueError) as context:
            self.batch_file.create_batch_file(requests)
        self.assertIn("Missing required field 'url'", str(context.exception))
        
        # Invalid method
        requests = [{
            "custom_id": "request-1",
            "method": "GET",  # Should be POST
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o"}
        }]
        
        with self.assertRaises(ValueError) as context:
            self.batch_file.create_batch_file(requests)
        self.assertIn("Only POST method is supported", str(context.exception))
        
        # Invalid endpoint
        requests = [{
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/invalid",
            "body": {"model": "gpt-4o"}
        }]
        
        with self.assertRaises(ValueError) as context:
            self.batch_file.create_batch_file(requests)
        self.assertIn("Unsupported endpoint", str(context.exception))
    
    def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.batch_file.read_batch_file("nonexistent-file-id")
    
    def test_create_output_file(self):
        """Test creating an output file."""
        results = [
            {
                "custom_id": "request-1",
                "response": {
                    "status_code": 200,
                    "body": {"result": "success"}
                }
            }
        ]
        
        output_info = self.batch_file.create_output_file("batch-123", results)
        
        self.assertEqual(output_info["purpose"], "batch_output")
        self.assertEqual(output_info["batch_id"], "batch-123")
        self.assertIn("_output.jsonl", output_info["filename"])


class TestBatch(unittest.TestCase):
    """Test cases for Batch API."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        # Patch the storage path
        self.patcher = patch.object(
            Batch, '_storage_path',
            Path(self.temp_dir)
        )
        self.patcher.start()
        # Clear batch storage
        Batch._batches.clear()
    
    def tearDown(self):
        """Clean up test environment."""
        self.patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('nerif.batch.batch.SimpleChatModel')
    @patch('nerif.batch.batch.asyncio.create_task')
    def test_create_batch(self, mock_create_task, mock_chat_model):
        """Test creating a batch."""
        # Create test input file
        batch_file = BatchFile(self.temp_dir)
        requests = [{
            "custom_id": "test-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        }]
        file_info = batch_file.create_batch_file(requests)
        
        # Create batch
        batch = Batch.create(
            input_file_id=file_info["id"],
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"test": "true"}
        )
        
        # Verify batch structure
        self.assertEqual(batch["object"], "batch")
        self.assertEqual(batch["endpoint"], "/v1/chat/completions")
        self.assertEqual(batch["input_file_id"], file_info["id"])
        self.assertEqual(batch["completion_window"], "24h")
        self.assertEqual(batch["status"], BatchStatus.VALIDATING)
        self.assertEqual(batch["metadata"]["test"], "true")
        self.assertEqual(batch["request_counts"]["total"], 1)
        self.assertIsNotNone(batch["created_at"])
        self.assertIsNotNone(batch["expires_at"])
        
        # Verify async task was created
        mock_create_task.assert_called_once()
    
    def test_create_batch_invalid_endpoint(self):
        """Test creating batch with invalid endpoint."""
        with self.assertRaises(ValueError) as context:
            Batch.create(
                input_file_id="file-123",
                endpoint="/v1/invalid",
                completion_window="24h"
            )
        self.assertIn("Unsupported endpoint", str(context.exception))
    
    def test_create_batch_invalid_window(self):
        """Test creating batch with invalid completion window."""
        with self.assertRaises(ValueError) as context:
            Batch.create(
                input_file_id="file-123",
                endpoint="/v1/chat/completions",
                completion_window="48h"
            )
        self.assertIn("Only '24h' completion window", str(context.exception))
    
    def test_create_batch_too_much_metadata(self):
        """Test creating batch with too much metadata."""
        metadata = {f"key{i}": f"value{i}" for i in range(20)}
        
        with self.assertRaises(ValueError) as context:
            Batch.create(
                input_file_id="file-123",
                endpoint="/v1/chat/completions",
                metadata=metadata
            )
        self.assertIn("16 key-value pairs", str(context.exception))
    
    @patch('nerif.batch.batch.asyncio.create_task')
    def test_retrieve_batch(self, mock_create_task):
        """Test retrieving a batch."""
        # Create a batch first
        batch_file = BatchFile(self.temp_dir)
        requests = [{
            "custom_id": "test-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o", "messages": []}
        }]
        file_info = batch_file.create_batch_file(requests)
        
        created_batch = Batch.create(
            input_file_id=file_info["id"],
            endpoint="/v1/chat/completions"
        )
        
        # Retrieve the batch
        retrieved_batch = Batch.retrieve(created_batch["id"])
        
        self.assertEqual(retrieved_batch["id"], created_batch["id"])
        self.assertEqual(retrieved_batch["status"], created_batch["status"])
        self.assertEqual(retrieved_batch["endpoint"], created_batch["endpoint"])
    
    def test_retrieve_nonexistent_batch(self):
        """Test retrieving a batch that doesn't exist."""
        with self.assertRaises(ValueError) as context:
            Batch.retrieve("nonexistent-batch-id")
        self.assertIn("not found", str(context.exception))
    
    @patch('nerif.batch.batch.asyncio.create_task')
    def test_cancel_batch(self, mock_create_task):
        """Test cancelling a batch."""
        # Create a batch
        batch_file = BatchFile(self.temp_dir)
        requests = [{
            "custom_id": "test-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o", "messages": []}
        }]
        file_info = batch_file.create_batch_file(requests)
        
        created_batch = Batch.create(
            input_file_id=file_info["id"],
            endpoint="/v1/chat/completions"
        )
        
        # Cancel the batch
        cancelled_batch = Batch.cancel(created_batch["id"])
        
        self.assertEqual(cancelled_batch["status"], BatchStatus.CANCELLED)
        self.assertIsNotNone(cancelled_batch["cancelled_at"])
    
    @patch('nerif.batch.batch.asyncio.create_task')
    def test_list_batches(self, mock_create_task):
        """Test listing batches."""
        batch_file = BatchFile(self.temp_dir)
        
        # Create multiple batches
        batch_ids = []
        for i in range(3):
            requests = [{
                "custom_id": f"test-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-4o", "messages": []}
            }]
            file_info = batch_file.create_batch_file(requests)
            batch = Batch.create(
                input_file_id=file_info["id"],
                endpoint="/v1/chat/completions"
            )
            batch_ids.append(batch["id"])
            time.sleep(0.1)  # Ensure different timestamps
        
        # List all batches
        batch_list = Batch.list(limit=10)
        
        self.assertEqual(batch_list["object"], "list")
        self.assertEqual(len(batch_list["data"]), 3)
        self.assertFalse(batch_list["has_more"])
        
        # Test pagination
        batch_list = Batch.list(limit=2)
        self.assertEqual(len(batch_list["data"]), 2)
        self.assertTrue(batch_list["has_more"])
        
        # Test after cursor
        first_id = batch_list["data"][0]["id"]
        batch_list = Batch.list(after=first_id, limit=10)
        self.assertEqual(len(batch_list["data"]), 2)


class TestBatchProcessing(unittest.TestCase):
    """Test batch processing functionality."""
    
    @patch('nerif.batch.batch.SimpleChatModel')
    def test_process_single_chat(self, mock_chat_model):
        """Test processing a single chat request."""
        # Mock the chat model
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "Hello, I'm an AI assistant."
        mock_instance.messages = []
        mock_chat_model.return_value = mock_instance
        
        batch = Batch()
        request = {
            "custom_id": "test-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "temperature": 0.7,
                "max_tokens": 150
            }
        }
        
        result = batch._process_single_chat(request)
        
        # Verify result structure
        self.assertEqual(result["custom_id"], "test-1")
        self.assertEqual(result["response"]["status_code"], 200)
        self.assertIsNotNone(result["response"]["body"])
        self.assertEqual(
            result["response"]["body"]["choices"][0]["message"]["content"],
            "Hello, I'm an AI assistant."
        )
        self.assertIsNone(result["error"])
        
        # Verify model was called correctly
        mock_chat_model.assert_called_once_with(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=150
        )
    
    @patch('nerif.batch.batch.SimpleEmbeddingModel')
    def test_process_single_embedding(self, mock_embedding_model):
        """Test processing a single embedding request."""
        # Mock the embedding model
        mock_instance = MagicMock()
        mock_instance.embed.return_value = [0.1, 0.2, 0.3]
        mock_embedding_model.return_value = mock_instance
        
        batch = Batch()
        request = {
            "custom_id": "test-2",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": "text-embedding-3-small",
                "input": "Sample text"
            }
        }
        
        result = batch._process_single_embedding(request)
        
        # Verify result structure
        self.assertEqual(result["custom_id"], "test-2")
        self.assertEqual(result["response"]["status_code"], 200)
        self.assertEqual(
            result["response"]["body"]["data"][0]["embedding"],
            [0.1, 0.2, 0.3]
        )
        self.assertIsNone(result["error"])
        
        # Verify model was called
        mock_embedding_model.assert_called_once_with(
            model="text-embedding-3-small"
        )
        mock_instance.embed.assert_called_once_with("Sample text")
    
    def test_create_error_response(self):
        """Test creating an error response."""
        batch = Batch()
        request = {
            "custom_id": "test-error",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {}
        }
        exception = ValueError("Test error message")
        
        error_response = batch._create_error_response(request, exception)
        
        self.assertEqual(error_response["custom_id"], "test-error")
        self.assertEqual(error_response["response"]["status_code"], 500)
        self.assertIsNone(error_response["response"]["body"])
        self.assertEqual(error_response["error"]["message"], "Test error message")
        self.assertEqual(error_response["error"]["type"], "ValueError")
        self.assertEqual(error_response["error"]["code"], "internal_error")


if __name__ == "__main__":
    unittest.main()