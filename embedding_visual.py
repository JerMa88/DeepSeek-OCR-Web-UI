
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import seaborn as sns
from typing import Tuple, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# ============== DeepSeek-OCR Model Setup ==============
model_name = 'deepseek-ai/DeepSeek-OCR'
save_directory = './model'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    cache_dir=save_directory,
    _attn_implementation='flash_attention_2', 
    trust_remote_code=True, 
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)
print("Model architecture:", model)

# ============== Optical Compression Module ==============
class OpticalCompressor(nn.Module):
    """
    Implements DeepSeek-OCR style compression: text → 2D vision tokens → compressed representation
    Based on the paper's 16× compression approach
    """
    def __init__(
        self, 
        input_dim: int = 768,  # Standard text embedding dimension
        compression_ratio: int = 16,
        patch_size: int = 16,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.input_dim = input_dim
        self.compression_ratio = compression_ratio
        self.patch_size = patch_size
        
        # Calculate grid dimensions for 2D mapping
        self.grid_size = int(np.sqrt(input_dim))
        if self.grid_size ** 2 < input_dim:
            self.grid_size = int(np.ceil(np.sqrt(input_dim)))
        
        # Compression layers (mimicking DeepEncoder's approach)
        # Stage 1: Initial projection
        self.projection = nn.Linear(input_dim, self.grid_size * self.grid_size)
        
        # Stage 2: 2D Convolutional compression (like DeepEncoder's 16× compressor)
        self.conv_compressor = nn.Sequential(
            # First conv layer: kernel=3, stride=2, channels 256→512
            nn.Conv2d(1, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            
            # Second conv layer: kernel=3, stride=2, channels 512→1024
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim * 2),
            
            # Additional compression if needed
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim * 4),
        )
        
        # Calculate compressed dimensions
        compressed_grid = self.grid_size
        for _ in range(3):  # 3 conv layers with stride=2
            compressed_grid = (compressed_grid + 1) // 2
        self.compressed_dim = compressed_grid * compressed_grid * (hidden_dim * 4)
        
    def forward(self, text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress text embeddings to vision tokens
        Returns: (compressed_embedding, 2d_representation_for_visualization)
        """
        batch_size = text_embeddings.shape[0]
        
        # Project to square dimension
        projected = self.projection(text_embeddings)
        
        # Reshape to 2D grid
        vision_2d = projected.view(batch_size, 1, self.grid_size, self.grid_size)
        
        # Apply convolutional compression
        compressed_2d = self.conv_compressor(vision_2d)
        
        # Flatten for storage
        compressed_flat = compressed_2d.view(batch_size, -1)
        
        return compressed_flat, compressed_2d

class TextToOpticalPipeline:
    """Complete pipeline for text → optical compression visualization"""
    
    def __init__(self, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        
        # Text embedding model
        self.text_encoder = SentenceTransformer(embedding_model_name)
        
        # Optical compressor
        embedding_dim = self.text_encoder.get_sentence_embedding_dimension()
        self.compressor = OpticalCompressor(input_dim=embedding_dim).cuda()
        
    def process_text(self, text: str):
        """Process text through the full compression pipeline"""
        
        # Step 1: Generate text embeddings
        text_embedding = self.text_encoder.encode(text, convert_to_tensor=True)
        text_embedding = text_embedding.cuda().unsqueeze(0)
        
        # Step 2: Compress to optical representation
        with torch.no_grad():
            compressed_embedding, compressed_2d = self.compressor(text_embedding)
        
        return {
            'text': text,
            'text_embedding': text_embedding.cpu().numpy(),
            'compressed_embedding': compressed_embedding.cpu().numpy(),
            'compressed_2d': compressed_2d.cpu().numpy(),
            'compression_ratio': text_embedding.shape[-1] / compressed_embedding.shape[-1]
        }
    
    def visualize_compression(self, result: dict, save_path: str = './compression_viz.png'):
        """Visualize the complete compression pipeline"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('DeepSeek-OCR Style Optical Compression Pipeline', fontsize=16)
        
        # 1. Original text
        axes[0, 0].text(0.5, 0.5, result['text'][:500] + '...', 
                       ha='center', va='center', wrap=True, fontsize=10)
        axes[0, 0].set_title('1. Original Text')
        axes[0, 0].axis('off')
        
        # 2. Text embedding heatmap
        embedding_2d = result['text_embedding'].reshape(-1, 32)  # Reshape for visualization
        im1 = axes[0, 1].imshow(embedding_2d, cmap='viridis', aspect='auto')
        axes[0, 1].set_title(f'2. Text Embedding\n(dim={result["text_embedding"].shape[-1]})')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 3. Compressed embedding heatmap
        compressed_1d = result['compressed_embedding'].reshape(-1, 64)  # Reshape for visualization
        im2 = axes[0, 2].imshow(compressed_1d, cmap='coolwarm', aspect='auto')
        axes[0, 2].set_title(f'3. Compressed Embedding\n(dim={result["compressed_embedding"].shape[-1]})')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # 4. 2D vision representation (multiple channels)
        compressed_2d = result['compressed_2d'][0]  # Remove batch dimension
        # Average across channels for visualization
        vision_map = np.mean(compressed_2d, axis=0)
        im3 = axes[1, 0].imshow(vision_map, cmap='hot', interpolation='nearest')
        axes[1, 0].set_title(f'4. 2D Vision Tokens\n(Averaged across {compressed_2d.shape[0]} channels)')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 5. Channel visualization (first 4 channels)
        channel_viz = np.zeros((compressed_2d.shape[1] * 2, compressed_2d.shape[2] * 2))
        for i in range(min(4, compressed_2d.shape[0])):
            row = i // 2
            col = i % 2
            channel_viz[row*compressed_2d.shape[1]:(row+1)*compressed_2d.shape[1],
                       col*compressed_2d.shape[2]:(col+1)*compressed_2d.shape[2]] = compressed_2d[i]
        
        im4 = axes[1, 1].imshow(channel_viz, cmap='plasma', interpolation='nearest')
        axes[1, 1].set_title('5. First 4 Channels of Compressed 2D')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # 6. Compression statistics
        stats_text = f"""Compression Statistics:
        
Original Dimension: {result['text_embedding'].shape[-1]}
Compressed Dimension: {result['compressed_embedding'].shape[-1]}
Compression Ratio: {result['compression_ratio']:.2f}×
        
DeepSeek-OCR Target Ratios:
- <10×: 97% precision
- 10-12×: ~90% precision  
- 20×: ~60% precision

Current: {result['compression_ratio']:.2f}×"""
        
        axes[1, 2].text(0.1, 0.5, stats_text, ha='left', va='center', fontsize=11, 
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 2].set_title('6. Compression Analysis')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig

# ============== Main Execution ==============
def main():
    # Run DeepSeek-OCR on image
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    image_file = './images/rec_sys.png'
    output_path = './output/'
    
    print("Running DeepSeek-OCR inference...")
    ocr_result = model.infer(
        tokenizer, 
        prompt=prompt, 
        image_file=image_file, 
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
    
    # Initialize compression pipeline
    print("\nInitializing optical compression pipeline...")
    pipeline = TextToOpticalPipeline()
    
    # Process the OCR output through compression
    print("Compressing text to optical representation...")
    compression_result = pipeline.process_text(ocr_result)
    
    # Visualize the compression
    print("\nGenerating visualization...")
    pipeline.visualize_compression(compression_result, save_path='./optical_compression_demo.png')
    
    # Print compression details
    print("\n" + "=" * 80)
    print("COMPRESSION SUMMARY")
    print("=" * 80)
    print(f"Original text length: {len(ocr_result)} characters")
    print(f"Text embedding dimension: {compression_result['text_embedding'].shape[-1]}")
    print(f"Compressed embedding dimension: {compression_result['compressed_embedding'].shape[-1]}")
    print(f"Compression ratio: {compression_result['compression_ratio']:.2f}×")
    print(f"Storage reduction: {(1 - 1/compression_result['compression_ratio']) * 100:.1f}%")
    
    # Save compressed representation for later retrieval
    np.save('./compressed_vector.npy', compression_result['compressed_embedding'])
    print(f"\nCompressed vector saved to ./compressed_vector.npy")
    
    return compression_result

if __name__ == "__main__":
    result = main()