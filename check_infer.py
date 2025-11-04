#!/usr/bin/env python3

import torch
from transformers import AutoModel, AutoTokenizer
import inspect

def check_infer_signature():
    model_name = 'deepseek-ai/DeepSeek-OCR'
    save_directory = './model'
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=save_directory,
        trust_remote_code=True
    ).eval()
    
    print("Checking infer method signature...")
    sig = inspect.signature(model.infer)
    print(f"Signature: {sig}")
    
    # Try to get source code
    try:
        source = inspect.getsource(model.infer)
        print("Source code:")
        print(source[:1000])  # First 1000 chars
    except:
        print("Could not get source code")
    
    return model, tokenizer

if __name__ == "__main__":
    check_infer_signature()