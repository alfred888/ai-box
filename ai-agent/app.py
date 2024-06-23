from flask import Flask, request, jsonify
from functools import wraps
import json
from text_interaction import gpt4_text_interaction
from image_interaction import gpt4_image_interaction
from video_interaction import gpt4_video_interaction

app = Flask(__name__)

def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

config = load_config()
openai_api_key = config['openai_api_key']

# Token-based authentication
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-tokens')
        if not token or token != 'YOUR_SECRET_TOKEN':
            return jsonify({'message': 'Token is missing or invalid!'}), 403
        return f(*args, **kwargs)
    return decorated

@app.route('/text', methods=['POST'])
@token_required
def text_interaction():
    data = request.json
    prompt = data['prompt']
    response = gpt4_text_interaction(openai_api_key, prompt)
    return jsonify({'response': response})

@app.route('/image', methods=['POST'])
@token_required
def image_interaction():
    data = request.json
    image_path = data['image_path']
    response = gpt4_image_interaction(openai_api_key, image_path)
    return jsonify({'response': response})

@app.route('/video', methods=['POST'])
@token_required
def video_interaction():
    data = request.json
    video_path = data['video_path']
    prompt = data['prompt']
    response, _ = gpt4_video_interaction(openai_api_key, video_path, prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(ssl_context=('cert.pem', 'key.pem'))