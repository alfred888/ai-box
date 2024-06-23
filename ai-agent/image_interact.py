import openai
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests

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

def recognize_image(image_path):
    model = models.resnet50(pretrained=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = requests.get(LABELS_URL).json()
    label = labels[preds.item()]
    return label

def gpt4_image_interaction(api_key, image_path):
    label = recognize_image(image_path)
    prompt = f"The image contains a {label}. Please describe it in more detail."
    gpt_response = gpt4_text_interaction(api_key, prompt)
    return gpt_response