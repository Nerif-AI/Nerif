"""Example: Streaming responses for real-time output.

Demonstrates stream_chat() for token-by-token output, useful for
long-form generation and interactive applications.
"""

from nerif.model import SimpleChatModel

model = SimpleChatModel()

# Stream response - prints tokens as they arrive
print("Streaming response:")
for chunk in model.stream_chat("Write a short poem about Python programming."):
    print(chunk, end="", flush=True)
print()  # newline after stream

# Stream with conversation history
print("\nStreaming with history:")
for chunk in model.stream_chat("Now translate it to Japanese.", append=True):
    print(chunk, end="", flush=True)
print()
