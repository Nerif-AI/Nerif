import nerif.model as model
import nerif.utils.log as log

log.set_up_logging(out_file="sample.log", mode="w", std=True)

model = model.SimpleChatModel()

print(model.chat("What is the capital of the moon?"))
print(model.chat("What is the capital of the moon?", max_tokens=10))
