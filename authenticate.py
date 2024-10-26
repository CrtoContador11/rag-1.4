import os
from huggingface_hub import login
from dotenv import load_dotenv

def authenticate():
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not token:
        print("Error: HUGGINGFACE_TOKEN not found in .env file")
        return False
        
    try:
        print("Authenticating with Hugging Face...")
        login(token=token)
        print("Authentication successful!")
        return True
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
        print("\nPlease check:")
        print("1. Your HUGGINGFACE_TOKEN is valid")
        print("2. You have internet connection")
        print("3. You have access to the model")
        return False

if __name__ == "__main__":
    authenticate()