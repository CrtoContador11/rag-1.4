import os
from pathlib import Path
from src.interface.app import create_interface
from authenticate import authenticate

def setup_directories():
    base_dir = r"C:\Users\danie\OneDrive - SCP Ozona SL\Escritorio\RAG"
    directories = ['knowledge', 'db']
    
    for dir_name in directories:
        dir_path = Path(base_dir) / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    if not authenticate():
        print("Authentication failed. Please fix the issues above and try again.")
        exit(1)
        
    setup_directories()
    interface = create_interface()
    interface.launch(share=False)