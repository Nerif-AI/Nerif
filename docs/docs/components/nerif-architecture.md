---
sidebar_position: 1
---

# Nerif Architecture

This part, I will introduce core design part by part, here is a general map

![Nerif Architecture](img/arch.png)

Let me help improve the documentation to make it more polished and comprehensive:

### Nerif Model

Like other multi-agent frameworks, Nerif Model provides flexibility in utilizing various AI models, including multi-modal capabilities. Our framework supports models that can interact with external APIs and tools for enhanced functionality.

Currently, we support fundamental AI capabilities including:
- LLM chat models
- Vision models
- Embedding models

In upcoming releases, we plan to expand support for custom models and external API integrations.

### Nerif Core

The key distinction between `model` and `core` lies in their type system implementation. While LLM/VLM models typically generate natural language outputs that require complex post-processing, Nerif Core ensures the outputs are properly typed and immediately usable in your applications.

Our core functionality consists of six essential modules:

1. **Nerif**: Evaluates statements and returns boolean values (`True`/`False`)
2. **Nerification**: Validates statements with boolean responses (`True`/`False`)
3. **Nerif Match**: Takes a statement and a list as input, returning the index of the best-matching item
4. **Nerif Format**: Handles type conversion between different formats
5. **Nerif Json**: Structures outputs in JSON format according to specified requirements
6. **Nerif Log**: Provides comprehensive logging capabilities

### Nerif Flow

This feature will be available after the v1.0 release.