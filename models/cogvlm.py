import os
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM, LlamaTokenizer
import unicodedata

class CogVLMInferencer:
    def __init__(self, model_name='zai-org/cogvlm-chat-hf', device='cuda'):
        """
        Initialize CogVLM model and tokenizer.
        
        Args:
            model_name (str): HuggingFace model name
            device (str): Device to run model on ('cuda' or 'cpu')
        """
        self.device = device
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        print(f"Loading CogVLM tokenizer and model on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if self.device == 'cuda':
            self.model = self.model.to('cuda')
        
        self.model.eval()
        print("CogVLM model loaded successfully")
        
        # Create dummy image for text-only inference
        self._dummy_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
    
    def _remove_control_characters(self, s):
        """Remove control characters from string."""
        return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    
    def _prepare_image(self, image):
        """
        Convert various image formats to PIL Image.
        
        Args:
            image: Can be file path (str), bytes, or PIL Image
            
        Returns:
            PIL.Image: Processed image
        """
        try:
            # If image is a file path
            if isinstance(image, str):
                return Image.open(image).convert('RGB')
            
            # If image is bytes
            elif isinstance(image, bytes):
                return Image.open(BytesIO(image)).convert('RGB')
            
            # If image is already a PIL Image
            elif hasattr(image, 'save'):
                return image.convert('RGB')
            
            else:
                raise TypeError("Image must be a file path, bytes, or PIL Image")
                
        except Exception as e:
            print(f"Error preparing image: {e}")
            raise
    
    def infer(self, text, image=None, max_new_tokens=512, do_sample=False, num_beams=1):
        """
        Run inference on CogVLM model with text and optional image input.
        
        Args:
            text (str): Text prompt to send to model
            image (str, bytes, or PIL.Image, optional): Image input
            max_new_tokens (int): Maximum number of tokens to generate
            do_sample (bool): Whether to use sampling
            num_beams (int): Number of beams for beam search
            
        Returns:
            str: The model's response
        """
        try:
            # Prepare image - use dummy if no image provided
            if image is not None:
                pil_image = self._prepare_image(image)
            else:
                pil_image = self._dummy_image
            
            # Build conversation input
            conversation_data = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=text,
                history=[],
                images=[pil_image]
            )
            
            # Prepare inputs with proper tensor conversion
            inputs = {
                'input_ids': conversation_data['input_ids'].unsqueeze(0),
                'attention_mask': conversation_data['attention_mask'].unsqueeze(0),
                'token_type_ids': conversation_data['token_type_ids'].unsqueeze(0),
                'images': [[conversation_data['images'][0]]]
            }
            
            # Move to device and convert dtype if needed
            if self.device == 'cuda':
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                inputs['token_type_ids'] = inputs['token_type_ids'].cuda()
                inputs['images'] = [[inputs['images'][0][0].to(dtype=torch.bfloat16).cuda()]]
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    num_beams=num_beams
                )
                
                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                # Clean response
                final_response = self._remove_control_characters(response.strip())
                
                return final_response
                
        except Exception as e:
            print(f"Error in CogVLM inference: {e}")
            return f"Error: {str(e)}"
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()