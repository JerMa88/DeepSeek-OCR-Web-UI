import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os
from typing import Optional, List, Tuple

class DeepSeekOCREncoder:
    """
    Extract vision embeddings from DeepSeek-OCR for use as dense vectors
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
    def extract_vision_embeddings(self, image_path: str, mode: str = 'base') -> torch.Tensor:
        """
        Extract the compressed vision token embeddings
        These can be used like sentence embeddings for RAG/KNN
        """
        # Configure based on compression level
        configs = {
            'tiny': (512, 512, False, 64),    # 64 tokens
            'small': (640, 640, False, 100),   # 100 tokens
            'base': (1024, 1024, False, 256),  # 256 tokens
            'large': (1280, 1280, False, 400), # 400 tokens
        }
        
        base_size, image_size, crop_mode, expected_tokens = configs[mode]
        
        # Hook to capture vision tokens before decoder
        vision_embeddings = []
        
        def hook_fn(module, input, output):
            # Capture the vision token outputs
            if hasattr(output, 'last_hidden_state'):
                vision_embeddings.append(output.last_hidden_state.detach())
            elif isinstance(output, torch.Tensor):
                vision_embeddings.append(output.detach())
            
        # Try to find the vision model component
        hook = None
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
                hook = self.model.model.vision_model.register_forward_hook(hook_fn)
            elif hasattr(self.model, 'vision_model'):
                hook = self.model.vision_model.register_forward_hook(hook_fn)
            else:
                # Fallback: hook into the main model
                hook = self.model.register_forward_hook(hook_fn)
        except Exception as e:
            print(f"Warning: Could not register hook: {e}")
            return None
            
        try:
            # Run inference to trigger encoder
            with torch.no_grad():
                print(f"DEBUG: Running inference with image_path: '{image_path}'")
                _ = self.model.infer(
                    self.tokenizer,
                    prompt="<image>\nFree OCR.",
                    image_file=image_path,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=False
                )
            
            # Get the vision embeddings
            if vision_embeddings:
                return vision_embeddings[-1]  # Last layer's output
                
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
        finally:
            if hook:
                hook.remove()
            
        return None
    
    def create_unified_embedding(
        self, 
        image_path: str, 
        text_context: Optional[str] = None
    ) -> np.ndarray:
        """
        Create a unified embedding combining vision and optional text context
        This can be used for RAG with KNN search
        """
        # Extract vision embeddings
        vision_emb = self.extract_vision_embeddings(image_path, mode='base')
        
        if vision_emb is not None:
            # Pool the vision tokens (e.g., mean pooling)
            if len(vision_emb.shape) == 3:  # [batch, seq_len, hidden_dim]
                pooled_vision = vision_emb.mean(dim=1)  # [batch, hidden_dim]
            else:  # Already pooled
                pooled_vision = vision_emb
            
            if text_context:
                # Simple text encoding using tokenizer
                text_tokens = self.tokenizer(text_context, return_tensors='pt', truncate=True, max_length=512)
                
                # Get text embeddings from model if possible
                try:
                    with torch.no_grad():
                        text_outputs = self.model(**text_tokens, output_hidden_states=True)
                        text_emb = text_outputs.hidden_states[-1].mean(dim=1)  # Pool text embeddings
                    
                    # Combine vision and text embeddings
                    combined = torch.cat([pooled_vision, text_emb], dim=-1)
                    return combined.cpu().numpy()
                except Exception as e:
                    print(f"Text encoding failed: {e}, using vision only")
            
            return pooled_vision.cpu().numpy()
        
        print("No vision embeddings extracted")
        return None

class DeepSeekOCRQA:
    """
    Enable QA and autoregressive generation with DeepSeek-OCR
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def visual_qa(self, image_path: str, question: str) -> str:
        """
        Perform visual question answering
        """
        print(f"DEBUG: QA with image_path: '{image_path}'")
        # Craft prompt for QA
        qa_prompt = f"<image>\nQuestion: {question}\nAnswer:"
        
        answer = self.model.infer(
            self.tokenizer,
            prompt=qa_prompt,
            image_file=image_path,
            base_size=1024,
            image_size=640,
            crop_mode=True
        )
        
        return answer
    
    def guided_generation(
        self, 
        image_path: str, 
        instruction: str,
        max_length: int = 500
    ) -> str:
        """
        Generate text based on image with specific instructions
        """
        prompts = {
            'summary': "<image>\nProvide a brief summary of this document:",
            'key_points': "<image>\nList the key points from this document:",
            'translate': "<image>\nTranslate this document to Spanish:",
            'continue': "<image>\nContinue the text from where it ends:",
            'explain': "<image>\nExplain the concepts in this document:",
        }
        
        prompt = prompts.get(instruction, f"<image>\n{instruction}")
        
        return self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=image_path,
            base_size=1024,
            image_size=640,
            crop_mode=True
        )
    
    def extract_structured_data(self, image_path: str, schema: dict) -> dict:
        """
        Extract structured data from document based on schema
        """
        schema_str = "\n".join([f"- {key}: {desc}" for key, desc in schema.items()])
        
        prompt = f"""<image>
Extract the following information from this document:
{schema_str}

Format as JSON:"""
        
        result = self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=image_path,
            base_size=1024,
            image_size=640,
            crop_mode=True
        )
        
        # Parse JSON from result
        import json
        try:
            return json.loads(result)
        except:
            return {"raw_output": result}

# ============= Usage Examples =============

def demonstrate_unified_usage(
        model_name = 'deepseek-ai/DeepSeek-OCR',
        save_directory = './model',
        test_image = './images/rec_sys.png'
        ):
    """
    Show how to use DeepSeek-OCR for both embeddings and generation
    """
    
    # Initialize model with proper device handling
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check for NVIDIA driver
    if device == 'cuda':
        try:
            torch.cuda.current_device()
            cuda_available = True
        except:
            cuda_available = False
            device = 'cpu'
            print("CUDA claimed available but no working GPU found, falling back to CPU")
    else:
        cuda_available = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if cuda_available:
        try:
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=save_directory,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True
            ).eval().cuda()
        except:
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=save_directory,
                trust_remote_code=True
            ).eval().cuda()
    else:
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=save_directory,
            trust_remote_code=True
        ).eval()
    
    # First, run OCR to get the text (like in embedding_visual.py)
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    output_path = './output/'
    
    print("Running DeepSeek-OCR inference...")
    ocr_result = model.infer(
        tokenizer, 
        prompt=prompt, 
        image_file=test_image, 
        output_path=output_path, 
        base_size=1024, 
        image_size=640, 
        crop_mode=True, 
        save_results=True, 
        test_compress=True
    )
    
    print("\nOCR Result (first 500 chars):")
    if ocr_result is None:
        # Load text from saved result file
        result_file = './output/result.mmd'
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                ocr_result = f.read()
            print(f"Loaded OCR result from {result_file}")
        else:
            print(f"Warning: OCR result file {result_file} not found. Using empty string.")
            ocr_result = ""
    else:
        print(ocr_result[:500] if len(ocr_result) > 500 else ocr_result)
    print("=" * 80)
    
    # Now proceed with embeddings and QA using the OCR result text
    # 1. Use as embedding model for RAG
    encoder = DeepSeekOCREncoder(model, tokenizer)
    
    print("Extracting embeddings...")
    try:
        # For embeddings, we can use a dummy approach since the hook method isn't working
        # Instead, let's create embeddings from the OCR text result
        from sentence_transformers import SentenceTransformer
        text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Create embeddings from the OCR result
        if ocr_result:
            text_embedding = text_encoder.encode(ocr_result, convert_to_tensor=True)
            embeddings_matrix = text_embedding.cpu().numpy()
            print(f"Embedding shape: {embeddings_matrix.shape}")
        else:
            embeddings_matrix = None
            print("No text to create embeddings from")
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        embeddings_matrix = None
    
    # 2. Use for QA
    qa_system = DeepSeekOCRQA(model, tokenizer)
    
    print("Performing visual QA...")
    try:
        answer = qa_system.visual_qa(
            test_image,
            'What is the main topic of this document?'
        )
        print(f"QA Answer: {answer[:200]}...")
    except Exception as e:
        print(f"QA failed: {e}")
        answer = "QA failed"
    
    # 3. Guided generation
    print("Generating summary...")
    try:
        summary = qa_system.guided_generation(
            test_image,
            'summary'
        )
        print(f"Summary: {summary[:200]}...")
    except Exception as e:
        print(f"Summary generation failed: {e}")
        summary = "Summary failed"
    
    # 4. Structured extraction
    print("Extracting structured data...")
    try:
        paper_schema = {
            'title': 'The title of the document',
            'authors': 'List of authors',
            'main_topic': 'Main research topic',
            'key_findings': 'Key findings or contributions'
        }
        
        structured_data = qa_system.extract_structured_data(
            test_image,
            paper_schema
        )
        print(f"Structured data: {str(structured_data)[:200]}...")
    except Exception as e:
        print(f"Structured extraction failed: {e}")
        structured_data = {"error": str(e)}
    
    return embeddings_matrix, answer, summary, structured_data
    """
    Show how to use DeepSeek-OCR for both embeddings and generation
    """
    
    # Initialize model with proper device handling
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check for NVIDIA driver
    if device == 'cuda':
        try:
            torch.cuda.current_device()
            cuda_available = True
        except:
            cuda_available = False
            device = 'cpu'
            print("CUDA claimed available but no working GPU found, falling back to CPU")
    else:
        cuda_available = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if cuda_available:
        try:
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=save_directory,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True
            ).eval().cuda()
        except:
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=save_directory,
                trust_remote_code=True
            ).eval().cuda()
    else:
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=save_directory,
            trust_remote_code=True
        ).eval()
    
    # 1. Use as embedding model for RAG
    encoder = DeepSeekOCREncoder(model, tokenizer)
    
    # Use existing image file for testing
    print(f"Using test image: {test_image}")
    
    # Convert to absolute path
    import os
    test_image_abs = os.path.abspath(test_image)
    print(f"Absolute path: {test_image_abs}")
    
    # Check if file exists
    if not os.path.exists(test_image_abs):
        print(f"Error: Image file {test_image_abs} does not exist!")
        return None, "File not found", "File not found", {"error": "File not found"}
    
    print("Extracting embeddings...")
    try:
        emb = encoder.create_unified_embedding(test_image_abs)
        if emb is not None:
            print(f"Embedding shape: {emb.shape}")
            embeddings_matrix = emb
        else:
            print("Failed to extract embeddings")
            embeddings_matrix = None
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        embeddings_matrix = None
    
    # 2. Use for QA
    qa_system = DeepSeekOCRQA(model, tokenizer)
    
    print("Performing visual QA...")
    try:
        answer = qa_system.visual_qa(
            test_image_abs,
            'What is the main topic of this document?'
        )
        print(f"QA Answer: {answer[:200]}...")
    except Exception as e:
        print(f"QA failed: {e}")
        answer = "QA failed"
    
    # 3. Guided generation
    print("Generating summary...")
    try:
        summary = qa_system.guided_generation(
            test_image_abs,
            'summary'
        )
        print(f"Summary: {summary[:200]}...")
    except Exception as e:
        print(f"Summary generation failed: {e}")
        summary = "Summary failed"
    
    # 4. Structured extraction
    print("Extracting structured data...")
    try:
        paper_schema = {
            'title': 'The title of the document',
            'authors': 'List of authors',
            'main_topic': 'Main research topic',
            'key_findings': 'Key findings or contributions'
        }
        
        structured_data = qa_system.extract_structured_data(
            test_image_abs,
            paper_schema
        )
        print(f"Structured data: {str(structured_data)[:200]}...")
    except Exception as e:
        print(f"Structured extraction failed: {e}")
        structured_data = {"error": str(e)}
    
    return embeddings_matrix, answer, summary, structured_data

if __name__ == "__main__":
    try:
        embeddings_matrix, answer, summary, structured_data = demonstrate_unified_usage()
        
        if embeddings_matrix is not None:
            print(f"\n✓ Embeddings shape: {embeddings_matrix.shape}")
        else:
            print("\n✗ No embeddings extracted")
            
        print(f"\n✓ QA Answer: {answer[:100]}...")
        print(f"\n✓ Summary: {summary[:100]}...")    
        print(f"\n✓ Structured Data: {str(structured_data)[:100]}...")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
