from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat-Int4", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
).eval()
import pandas as pd
import base64
from datasets import load_dataset
import os
import matplotlib
matplotlib.use('notebook') 
import io
import os
if 'MPLBACKEND' in os.environ:
    del os.environ['MPLBACKEND']
dataset = load_dataset("molvision/BBBP-V-SMILES-2")

from PIL import Image
import torch

image_folder = 'images'
os.makedirs(image_folder, exist_ok=True)

df = pd.DataFrame(dataset['train'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

embeddings_list = []
root ='/content/'
for index, row in df.iterrows():
    if index ==1:
      break
    image = row['image']  
    query = row['Question']

    image_filename = f'image_{index}.png'
    image_path = os.path.join(image_folder, image_filename)
    image.save(image_path) 

    inputs = tokenizer(query, return_tensors="pt").to(device)


    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        embeddings = outputs.hidden_states[-1] 

        embeddings_list.append(embeddings.cpu().numpy())


        response = model.generate(**inputs)
        decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)

        print(decoded_response)
        df.at[index, 'predictions'] = decoded_response

# Save the predictions to a CSV file
df.to_csv(root +'Results/Qwen_new.csv', index=False)

