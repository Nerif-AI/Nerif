
from nerif.nerif_agent import *
from nerif.nerif_core import *

if __name__ == "__main__":
    vision_model = VisionAgent(model="openrouter/openai/gpt-4o-2024-08-06")
    vision_model.append_message(
        nerif_agent.MessageType.IMAGE_URL,
        "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png",
    )
    vision_model.append_message(nerif_agent.MessageType.TEXT, "what is in this image?")
    response = vision_model.chat()
    print(response)
