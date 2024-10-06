import nerif

nerif.set_up_logging(out_file="sample.log", mode="w", std=True)

agent = nerif.nerif_agent.SimpleChatAgent()

print(agent.chat("What is the capital of the moon?"))
print(agent.chat("What is the capital of the moon?", max_tokens=10))
