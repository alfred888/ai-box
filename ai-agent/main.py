import json
from text_interact import gpt4_text_interaction
from image_interact import gpt4_image_interaction
from video_interact import gpt4_video_interaction


def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config


def main():
    config = load_config()
    openai_api_key = config['openai_api_key']

    # Text interaction example
    text_prompt = "Tell me a story about a brave knight."
    text_response = gpt4_text_interaction(openai_api_key, text_prompt)
    print("Text Interaction Response:")
    print(text_response)

    # Image interaction example
    image_path = "input_image.jpg"
    image_response = gpt4_image_interaction(openai_api_key, image_path)
    print("Image Interaction Response:")
    print(image_response)

    # Video interaction example
    video_path = "input_video.mp4"
    video_prompt = "Summarize the content of this video."
    video_response = gpt4_video_interaction(openai_api_key, video_path, video_prompt)
    print("Video Interaction Response:")
    print(video_response)


if __name__ == "__main__":

    main()
