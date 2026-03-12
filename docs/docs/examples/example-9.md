# Example 9: File Logging

```python
from nerif.model import SimpleChatModel
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logger
logger = logging.getLogger(__name__)

# Setup file logging
file_handler = logging.FileHandler('nerif_app.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(file_handler)

# Use with models
agent = SimpleChatModel()

logger.info("Starting AI conversation")
response = agent.chat("What is artificial intelligence?")
logger.info(f"AI Response: {response[:50]}...")  # Log first 50 chars

# Different log levels
logger.debug("Detailed debugging information")
logger.warning("This might cause issues")
logger.error("An error occurred")
logger.critical("System critical error!")

# Log with context
try:
    # Some operation
    result = agent.chat("Complex query")
    logger.info(f"Successfully processed query")
except Exception as e:
    logger.error(f"Failed to process query: {e}")
```