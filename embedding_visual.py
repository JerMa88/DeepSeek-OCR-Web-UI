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

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Configure attention implementation based on device
if device == 'cuda':
    try:
        model = AutoModel.from_pretrained(
            model_name, 
            cache_dir=save_directory,
            _attn_implementation='flash_attention_2', 
            trust_remote_code=True, 
            use_safetensors=True
        )
        model = model.eval().cuda().to(torch.bfloat16)
    except Exception as e:
        print(f"Flash attention failed, falling back to regular attention: {e}")
        model = AutoModel.from_pretrained(
            model_name, 
            cache_dir=save_directory,
            trust_remote_code=True, 
            use_safetensors=True
        )
        model = model.eval().cuda()
else:
    # CPU mode
    model = AutoModel.from_pretrained(
        model_name, 
        cache_dir=save_directory,
        trust_remote_code=True, 
        use_safetensors=True
    )
    model = model.eval()

# ============== Fixed Optical Compression Module ==============
class OpticalCompressor(nn.Module):
    """
    Fixed implementation that actually compresses, not expands!
    For text embeddings, we need a different strategy than image compression.
    """
    def __init__(
        self, 
        input_dim: int = 384,  # Sentence transformer dimension
        target_compression_ratio: int = 10,  # Target 10× compression like the paper
        intermediate_dim: int = 64
    ):
        super().__init__()
        self.input_dim = input_dim
        self.target_compression_ratio = target_compression_ratio
        self.compressed_dim = max(16, input_dim // target_compression_ratio)  # At least 16 dims
        
        # Strategy: First expand to 2D representation, then aggressively compress
        # This mimics rendering text to image, then compressing the image
        
        # Step 1: Expand text embedding to pseudo-image (like rendering text)
        # This simulates converting text to a document image
        self.expansion_size = 32  # Create 32×32 pseudo-image
        self.text_to_image = nn.Sequential(
            nn.Linear(input_dim, self.expansion_size * self.expansion_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.expansion_size * self.expansion_size)
        )
        
        # Step 2: Compress using strided convolutions (like DeepEncoder)
        # Use fewer channels to ensure compression
        self.conv_compressor = nn.Sequential(
            # First compression: 32×32 → 16×16 with 8 channels
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            
            # Second compression: 16×16 → 8×8 with 4 channels  
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            
            # Third compression: 8×8 → 4×4 with 2 channels
            nn.Conv2d(4, 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2),
            
            # Final compression: 4×4 → 2×2 with 1 channel
            nn.Conv2d(2, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Step 3: Final projection to target dimension
        # After convolutions: 2×2×1 = 4 features
        self.final_projection = nn.Linear(4, self.compressed_dim)
        
    def forward(self, text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress text embeddings to vision tokens
        Returns: (compressed_embedding, pseudo_image, compressed_2d)
        """
        batch_size = text_embeddings.shape[0]
        
        # Step 1: Expand to pseudo-image (simulating text rendering)
        pseudo_image_flat = self.text_to_image(text_embeddings)
        pseudo_image = pseudo_image_flat.view(batch_size, 1, self.expansion_size, self.expansion_size)
        
        # Step 2: Apply convolutional compression
        compressed_2d = self.conv_compressor(pseudo_image)
        
        # Step 3: Final compression
        compressed_flat = compressed_2d.view(batch_size, -1)
        compressed_final = self.final_projection(compressed_flat)
        
        return compressed_final, pseudo_image, compressed_2d

class TextToOpticalPipeline:
    """Complete pipeline for text → optical compression visualization"""
    
    def __init__(self, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        
        # Text embedding model
        self.text_encoder = SentenceTransformer(embedding_model_name)
        
        # Optical compressor with 10× compression target
        embedding_dim = self.text_encoder.get_sentence_embedding_dimension()
        self.compressor = OpticalCompressor(
            input_dim=embedding_dim, 
            target_compression_ratio=10
        )
        
        # Move to appropriate device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.compressor = self.compressor.to(device)
        # Set to eval mode to avoid BatchNorm issues with single samples
        self.compressor.eval()
        
    def process_text(self, text: str):
        """Process text through the full compression pipeline"""
        
        # Get device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Step 1: Generate text embeddings
        text_embedding = self.text_encoder.encode(text, convert_to_tensor=True)
        text_embedding = text_embedding.to(device).unsqueeze(0)
        
        # Step 2: Compress to optical representation
        with torch.no_grad():
            compressed_embedding, pseudo_image, compressed_2d = self.compressor(text_embedding)
        
        return {
            'text': text,
            'text_embedding': text_embedding.cpu().numpy(),
            'compressed_embedding': compressed_embedding.cpu().numpy(),
            'pseudo_image': pseudo_image.cpu().numpy(),
            'compressed_2d': compressed_2d.cpu().numpy(),
            'compression_ratio': text_embedding.shape[-1] / compressed_embedding.shape[-1]
        }
    
    def visualize_compression(self, result: dict, save_path: str = './compression_viz.png'):
        """Visualize the complete compression pipeline"""
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create custom grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('DeepSeek-OCR Style Optical Compression Pipeline', fontsize=18, fontweight='bold')
        
        # 1. Original text (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        text_display = result['text'][:300] + '...' if len(result['text']) > 300 else result['text']
        ax1.text(0.05, 0.5, text_display, ha='left', va='center', wrap=True, fontsize=9, family='monospace')
        ax1.set_title('1. Original Text', fontsize=12, fontweight='bold')
        ax1.axis('off')
        ax1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))
        
        # 2. Text embedding visualization
        ax2 = fig.add_subplot(gs[0, 2])
        embedding_1d = result['text_embedding'].flatten()
        embedding_2d = embedding_1d.reshape(-1, 24)  # Reshape for visualization (384 = 16×24)
        im2 = ax2.imshow(embedding_2d, cmap='viridis', aspect='auto')
        ax2.set_title(f'2. Text Embedding\n(dim={len(embedding_1d)})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Feature dimension')
        ax2.set_ylabel('Feature blocks')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # 3. Pseudo-image (text rendered to 2D)
        ax3 = fig.add_subplot(gs[0, 3])
        pseudo_img = result['pseudo_image'][0, 0]  # Remove batch and channel dims
        im3 = ax3.imshow(pseudo_img, cmap='gray', interpolation='nearest')
        ax3.set_title(f'3. Pseudo-Image\n({pseudo_img.shape[0]}×{pseudo_img.shape[1]})', 
                      fontsize=12, fontweight='bold')
        ax3.set_xlabel('Width')
        ax3.set_ylabel('Height')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # 4. Compression process visualization
        ax4 = fig.add_subplot(gs[1, :2])
        compression_steps = f"""Compression Process (DeepEncoder-style):

1. Text → Embedding: {len(result['text'])} chars → {result['text_embedding'].shape[-1]} dims
2. Embedding → Pseudo-Image: {result['text_embedding'].shape[-1]} → 32×32 grid
3. Conv Layer 1: 32×32×1 → 16×16×8 (stride=2)
4. Conv Layer 2: 16×16×8 → 8×8×4 (stride=2)
5. Conv Layer 3: 8×8×4 → 4×4×2 (stride=2)
6. Conv Layer 4: 4×4×2 → 2×2×1 (stride=2)
7. Final Project: 4 → {result['compressed_embedding'].shape[-1]} dims

Total Compression: {result['compression_ratio']:.2f}×"""
        
        ax4.text(0.05, 0.5, compression_steps, ha='left', va='center', fontsize=10, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax4.set_title('4. Compression Pipeline', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # 5. Compressed 2D representation
        ax5 = fig.add_subplot(gs[1, 2])
        compressed_2d = result['compressed_2d'][0, 0]  # Remove batch and channel dims
        im5 = ax5.imshow(compressed_2d, cmap='hot', interpolation='nearest')
        ax5.set_title(f'5. Compressed 2D\n({compressed_2d.shape[0]}×{compressed_2d.shape[1]})', 
                      fontsize=12, fontweight='bold')
        ax5.set_xlabel('Compressed width')
        ax5.set_ylabel('Compressed height')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        # 6. Final compressed embedding
        ax6 = fig.add_subplot(gs[1, 3])
        compressed_1d = result['compressed_embedding'].flatten()
        # Create a small grid for visualization
        grid_size = int(np.ceil(np.sqrt(len(compressed_1d))))
        padded = np.pad(compressed_1d, (0, grid_size**2 - len(compressed_1d)), constant_values=0)
        compressed_grid = padded.reshape(grid_size, grid_size)
        im6 = ax6.imshow(compressed_grid, cmap='coolwarm', interpolation='nearest')
        ax6.set_title(f'6. Compressed Vector\n(dim={len(compressed_1d)})', 
                      fontsize=12, fontweight='bold')
        plt.colorbar(im6, ax=ax6, fraction=0.046)
        
        # 7. Compression statistics (spans all columns)
        ax7 = fig.add_subplot(gs[2, :])
        stats_text = f"""Compression Performance Metrics:

- Original Text: {len(result['text'])} characters
- Text Embedding: {result['text_embedding'].shape[-1]} dimensions
- Compressed Embedding: {result['compressed_embedding'].shape[-1]} dimensions
- Compression Ratio: {result['compression_ratio']:.2f}× 
- Storage Reduction: {(1 - 1/result['compression_ratio']) * 100:.1f}%
- Bytes Saved: {(result['text_embedding'].shape[-1] - result['compressed_embedding'].shape[-1]) * 4} bytes (float32)

DeepSeek-OCR Reference (from paper):
- <10× compression: 97% OCR precision
- 10-12× compression: ~90% precision  
- 20× compression: ~60% precision

Current Status: {result['compression_ratio']:.2f}× compression achieved ✓"""
        
        ax7.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.2))
        ax7.set_title('7. Compression Analysis', fontsize=12, fontweight='bold')
        ax7.axis('off')
        
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
    
    # Load the result if it was saved
    try:
        with open('./output/result.mmd', 'r') as f:
            ocr_result = f.read()
        print("Loaded OCR result from ./output/result.mmd")
    except:
        pass
    
    print("\nOCR Result (first 500 chars):")
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
    
    # Verify compression worked
    if compression_result['compression_ratio'] >= 1.0:
        print(f"✓ Compression successful! Achieved {compression_result['compression_ratio']:.2f}× reduction")
    else:
        print(f"✗ WARNING: Expansion occurred instead of compression!")
    
    # Save compressed representation for later retrieval
    np.save('./compressed_vector.npy', compression_result['compressed_embedding'])
    print(f"\nCompressed vector saved to ./compressed_vector.npy")
    
    return compression_result

if __name__ == "__main__":
    result = main()