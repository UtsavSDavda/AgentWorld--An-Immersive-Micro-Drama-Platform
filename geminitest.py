import os
from videoprompts import *
from dotenv import load_dotenv
import time
from google import genai
from google.genai import types

load_dotenv()

gemini = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=gemini)

prompt = TROLL_DUNGEON
model_id = "veo-3.0-fast-generate-001"

print("Generating video...")

operation = client.models.generate_videos(
    model=model_id,
    prompt=prompt,
    config=types.GenerateVideosConfig(
        number_of_videos=1,
        aspect_ratio="16:9"
    )
)

while not operation.done:
    print("Waiting for video to finish processing...")
    time.sleep(5)
    operation = client.operations.get(operation)

if operation.result:
    generated_video = operation.result.generated_videos[0]
    print("Video generated successfully!")
    client.files.download(file=generated_video.video)
    generated_video.video.save("my_generated_video.mp4")
    print("Saved to my_generated_video.mp4")
else:
    print("Video generation failed.")