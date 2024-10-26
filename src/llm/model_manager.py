from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os
import sys

class LLMManager:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1"):
        load_dotenv()
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        
        if not self.token:
            print("Error: HUGGINGFACE_TOKEN not found in .env file")
            sys.exit(1)
            
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading model {model_name} on {self.device}...")
            print("Authenticating with Hugging Face...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=self.token,
                use_auth_token=self.token
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=self.token,
                use_auth_token=self.token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("\nPossible solutions:")
            print("1. Make sure your HUGGINGFACE_TOKEN is valid")
            print("2. Run 'huggingface-cli login' with your token")
            print("3. Check if you have access to the model")
            sys.exit(1)
    
    def generate_response(self, prompt: str, max_length: int = 512):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)