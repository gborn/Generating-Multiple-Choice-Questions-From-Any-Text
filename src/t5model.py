
import requests
import json
import os

API_URL = "https://api-inference.huggingface.co/models/mrm8488/t5-base-finetuned-question-generation-ap"
API_TOKEN = os.getenv('API_TOKEN')

headers = {"Authorization": f"{API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    output =  json.loads(response.content.decode("utf-8"))
    return output

# ping model to wake it up when this package is imported
print(query("answer: Manuel context: Manuel has created RuPERTa-base with the support of HF-Transformers and Google"))