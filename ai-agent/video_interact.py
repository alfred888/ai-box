import openai
from moviepy.editor import VideoFileClip

def gpt4_text_interaction(api_key, prompt):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

def gpt4_video_interaction(api_key, video_path, prompt):
    video = VideoFileClip(video_path)
    duration = video.duration
    if duration > 30:
        video = video.subclip(0, 30)
    output_path = "processed_video.mp4"
    video.write_videofile(output_path, codec='libx264')
    extracted_text = "Extracted text from video"
    gpt_response = gpt4_text_interaction(api_key, extracted_text + " " + prompt)
    return gpt_response, output_path