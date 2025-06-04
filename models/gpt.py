import os
import requests
from io import BytesIO
from PIL import Image
import base64
import json

class GPTInferencer:
   def __init__(self):
       self.api_key = os.environ.get("OPENAI_API_KEY")
       if not self.api_key:
           raise ValueError("OpenAI API key not found in environment variables")
       
       self.api_url = "https://api.openai.com/v1/chat/completions"
       self.headers = {
           "Authorization": f"Bearer {self.api_key}",
           "Content-Type": "application/json"
       }
       
   def _encode_image(self, image_path):
       """Encode image to base64."""
       try:
           with open(image_path, "rb") as image_file:
               return base64.b64encode(image_file.read()).decode('utf-8')
       except Exception as e:
           print(f"Error encoding image: {e}")
           return None
           
   def _encode_image_from_bytes(self, image_bytes):
       """Encode image from bytes to base64."""
       try:
           return base64.b64encode(image_bytes).decode('utf-8')
       except Exception as e:
           print(f"Error encoding image bytes: {e}")
           return None
   
   def infer(self, text, image=None, model="gpt-4o"):
       """
       Run inference on GPT model with text and optional image input.
       
       Args:
           text (str): Text prompt to send to model
           image (str or bytes, optional): Image path or bytes to include
           model (str): Model name ("gpt-4o" or "gpt-4v")
           
       Returns:
           str: The model's response
       """
       if model not in ["gpt-4o", "gpt-4v"]:
           raise ValueError("Model must be either 'gpt-4o' or 'gpt-4v'")
       
       messages = [{"role": "user", "content": []}]
       
       # Add text content
       messages[0]["content"].append({
           "type": "text",
           "text": text
       })
       
       # Add image content if provided
       if image:
           # If image is a path
           if isinstance(image, str):
               base64_image = self._encode_image(image)
           # If image is bytes
           elif isinstance(image, bytes):
               base64_image = self._encode_image_from_bytes(image)
           # If image is a PIL Image
           elif hasattr(image, 'save'):
               buffer = BytesIO()
               image.save(buffer, format="PNG")
               base64_image = self._encode_image_from_bytes(buffer.getvalue())
           else:
               raise TypeError("Image must be a file path, bytes, or PIL Image")
               
           if base64_image:
               messages[0]["content"].append({
                   "type": "image_url",
                   "image_url": {
                       "url": f"data:image/png;base64,{base64_image}"
                   }
               })
       
       # Prepare request payload
       payload = {
           "model": model,
           "messages": messages,
           "max_tokens": 1000
       }
       
       try:
           response = requests.post(self.api_url, headers=self.headers, json=payload)
           response.raise_for_status()
           result = response.json()
           return result["choices"][0]["message"]["content"]
       except Exception as e:
           print(f"Error in GPT inference: {e}")
           if hasattr(e, 'response') and hasattr(e.response, 'text'):
               print(f"Response error: {e.response.text}")
           return f"Error: {str(e)}"