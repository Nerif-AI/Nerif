import nerif.core.log as log
import nerif.agent.agent as agent

log.set_up_logging(out_file="sample.log", mode="w", std=True)

agent = agent.SimpleChatAgent()

print(agent.chat("What is the capital of the moon?"))
print(agent.chat("What is the capital of the moon?", max_tokens=10))
