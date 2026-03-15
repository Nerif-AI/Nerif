"""Example 27: Image Generation.

Demonstrates using Nerif's image generation module with
OpenAI DALL-E and Google Gemini backends.

Requires: OPENAI_API_KEY or GOOGLE_API_KEY environment variable
"""

from nerif.img_gen import ImageGenerationModel

# OpenAI DALL-E
model = ImageGenerationModel()

# Generate an image
# result = model.generate("A futuristic city skyline at sunset")
# for img in result.images:
#     print(f"Image URL: {img.url}")

# Google Gemini (NanoBananaModel)
# from nerif.img_gen import NanoBananaModel
# gemini_model = NanoBananaModel()
# result = gemini_model.generate("A cute robot reading a book")

print("ImageGenerationModel ready.")
print("Use ImageGenerationModel() for DALL-E or NanoBananaModel() for Gemini.")
