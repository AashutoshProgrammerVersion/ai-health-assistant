#!/usr/bin/env python3
"""
Check Gemini model specifications and token limits
"""

import os
import sys
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the client
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

def check_model_limits():
    """Check token limits for Flash models"""
    models_to_check = [
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite', 
        'gemini-2.0-flash',
        'gemini-2.0-flash-lite'
    ]
    
    print("Model Specifications:")
    print("=" * 80)
    
    for model_name in models_to_check:
        try:
            model_info = client.models.get(model=model_name)
            print(f"\n{model_name}:")
            print(f"  Input Token Limit: {model_info.input_token_limit:,}")
            print(f"  Output Token Limit: {model_info.output_token_limit:,}")
            print(f"  Display Name: {model_info.display_name}")
            print(f"  Description: {model_info.description}")
            
        except Exception as e:
            print(f"\n{model_name}: ERROR - {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_model_limits()