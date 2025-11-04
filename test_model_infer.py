#!/usr/bin/env python3

import torch
from transformers import AutoModel, AutoTokenizer

def test_model_infer():
    model_name = 'deepseek-ai/DeepSeek-OCR'
    save_directory = './model'
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=save_directory,
        trust_remote_code=True
    ).eval().cuda()
    
    image_path = './images/rec_sys.png'
    print(f"Testing with image: '{image_path}'")
    
    import os
    print(f"File exists: {os.path.exists(image_path)}")
    
    try:
        result = model.infer(
            tokenizer,
            prompt="<image>\nTest OCR.",
            image_file=image_path,
            base_size=512,
            image_size=512,
            crop_mode=False,
            save_results=False
        )
        print(f"Success! Result length: {len(result)}")
        print(f"First 100 chars: {result[:100]}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_infer()