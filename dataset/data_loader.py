# models/huggingface_datasets.py
from datasets import load_dataset

class HuggingFaceDataLoader:
   def __init__(self, dataset_name, split="train"):
       """
       Initialize the HuggingFace dataset loader.
       
       Args:
           dataset_name (str): Name of the HuggingFace dataset
           split (str): Dataset split to load (train, validation, test)
       """
       self.dataset_name = dataset_name
       self.split = split
       self.dataset = None
       
   def load_dataset(self):
       """Load the dataset from HuggingFace."""
       try:
           self.dataset = load_dataset(self.dataset_name, split=self.split)
           print(f"Successfully loaded {self.dataset_name}, split: {self.split}")
           print(f"Dataset contains {len(self.dataset)} samples")
           return self.dataset
       except Exception as e:
           print(f"Error loading dataset {self.dataset_name}: {e}")
           return None
   
   def get_sample(self, idx):
       """Get a specific sample from the dataset."""
       if self.dataset is None:
           self.load_dataset()
       
       if idx < 0 or idx >= len(self.dataset):
           raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.dataset)}")
           
       return self.dataset[idx]
   
   def get_all_samples(self):
       """Get all samples from the dataset."""
       if self.dataset is None:
           self.load_dataset()
           
       return self.dataset