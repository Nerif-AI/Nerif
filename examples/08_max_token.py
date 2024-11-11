import nerif

# case for ever specify

agent1 = nerif.agent.SimpleChatAgent()

print(agent1.chat("What is the capital of the moon?"))
agent1.counter

# case for specified in agent

agent2 = nerif.agent.SimpleChatAgent(max_tokens=5)

print(agent2.chat("What is the capital of the moon?"))
print(agent2.chat("What is the capital of the moon?", max_tokens=20))