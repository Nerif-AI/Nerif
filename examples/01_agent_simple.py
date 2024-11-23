import nerif

agent = nerif.model.SimpleChatModel()

print(agent.chat("What is the capital of the moon?"))
print(agent.chat("What is the capital of the moon?", max_tokens=10))

embedding_agent = nerif.model.SimpleEmbeddingModel()
print(embedding_agent.embed("What is the capital of the moon?"))

logits_agent = nerif.model.LogitsChatModel()
print(logits_agent.chat("What is the capital of the moon?"))
