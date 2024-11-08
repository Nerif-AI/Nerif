import nerif

agent = nerif.agent.SimpleChatAgent()

print(agent.chat("What is the capital of the moon?"))
print(agent.chat("What is the capital of the moon?", max_tokens=10))

embedding_agent = nerif.agent.SimpleEmbeddingAgent()
print(embedding_agent.encode("What is the capital of the moon?"))

logits_agent = nerif.agent.LogitsAgent()
print(logits_agent.chat("What is the capital of the moon?"))
