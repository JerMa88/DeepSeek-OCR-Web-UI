import gradio as gr
from transformers import AutoModel, AutoTokenizer
import torch
import os
import tempfile
from typing import List, Tuple
import base64
from PIL import Image, ImageDraw, ImageFont
import traceback
from i18n import i18n
import sys
from io import StringIO
import re
import random
import json

class OCRApp:
    """OCR Application with DeepSeek-OCR model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Model path - Can be configured via environment variable DEEPSEEK_OCR_MODEL_PATH
        # Download from: https://www.modelscope.cn/models/deepseek-ai/DeepSeek-OCR
        # or: https://huggingface.co/deepseek-ai/DeepSeek-OCR
        # Priority: Environment variable > Default value
        default_model_path = 'deepseek-ai/DeepSeek-OCR'
        self.model_path = os.environ.get('DEEPSEEK_OCR_MODEL_PATH', default_model_path)
        
        # Print model path info for debugging
        if os.environ.get('DEEPSEEK_OCR_MODEL_PATH'):
            print(f"Using model path from environment variable: {self.model_path}")
        else:
            print(f"Using default model path: {self.model_path}")
            print("To use a custom model path, set the environment variable: DEEPSEEK_OCR_MODEL_PATH")
        
    def set_model_path(self, path):
        """Set custom model path"""
        self.model_path = path
        self.model_loaded = False  # Reset model loading status
        
    def load_model(self):
        """Load OCR model"""
        if self.model_loaded:
            return
            
        try:
            # Set GPU device
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            
            print(f"Loading model from: {self.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                _attn_implementation='flash_attention_2', 
                trust_remote_code=True, 
                use_safetensors=True
            )
            self.model = self.model.eval().cuda().to(torch.bfloat16)
            self.model_loaded = True
            print(i18n.get('model_load_success'))
            
        except Exception as e:
            error_msg = i18n.get('model_load_failed', error=str(e))
            print(error_msg)
            traceback.print_exc()
            raise e
    
    def clean_ocr_result(self, raw_result: str) -> str:
        """Clean and format raw OCR result"""
        if not raw_result:
            return ""
        
        # Remove special markers and debug information
        lines = raw_result.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip debug information lines
            if line.startswith('BASE:') or line.startswith('PATCHES:') or line.startswith('====='):
                continue
            # Skip warning messages
            if 'attention layers' in line or 'position_ids' in line or 'position_embeddings' in line:
                continue
            # Skip empty lines
            if not line:
                continue
                
            # Process lines with special markers
            if '<|ref|>' in line and '<|/ref|>' in line:
                # Extract reference content
                ref_pattern = r'<\|ref\|>(.*?)<\|/ref\|>'
                refs = re.findall(ref_pattern, line)
                if refs:
                    cleaned_lines.extend(refs)
                    
                # Remove all markers and get remaining content
                cleaned_line = re.sub(r'<\|[^|]*\|>', '', line).strip()
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
            else:
                # Remove other markers
                cleaned_line = re.sub(r'<\|[^|]*\|>', '', line)
                cleaned_line = re.sub(r'\[\[[^\]]*\]\]', '', cleaned_line).strip()
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def extract_coordinates_from_text(self, text: str) -> List[Tuple[List[int], str]]:
        """Extract bounding box coordinates and associated text from OCR result"""
        coordinates_with_text = []
        
        # Pattern to match coordinates like [[609, 17, 656, 35]]
        coord_pattern = r'\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]'
        
        lines = text.split('\n')
        for line in lines:
            matches = re.finditer(coord_pattern, line)
            for match in matches:
                x1, y1, x2, y2 = map(int, match.groups())
                # Extract text associated with these coordinates (text after the coordinates on the same line)
                text_after_coords = line[match.end():].strip()
                if not text_after_coords:
                    # If no text after coordinates, try to get text before coordinates
                    text_before_coords = line[:match.start()].strip()
                    associated_text = text_before_coords if text_before_coords else f"Region {len(coordinates_with_text)+1}"
                else:
                    associated_text = text_after_coords
                
                coordinates_with_text.append(([x1, y1, x2, y2], associated_text))
        
        return coordinates_with_text
    
    def generate_color_palette(self, num_colors: int) -> List[str]:
        """Generate a palette of distinct colors for bounding boxes"""
        colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
            '#FFA500', '#800080', '#008000', '#FFC0CB', '#A52A2A', '#808080',
            '#000080', '#008080', '#800000', '#808000', '#C0C0C0', '#FF69B4',
            '#32CD32', '#FF4500', '#DA70D6', '#40E0D0', '#EE82EE', '#90EE90'
        ]
        
        # If we need more colors than predefined, generate random ones
        while len(colors) < num_colors:
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            if color not in colors:
                colors.append(color)
        
        return colors[:num_colors]
    
    def draw_bounding_boxes(self, image_path: str, coordinates_with_text: List[Tuple[List[int], str]]) -> str:
        """Draw colored bounding boxes on image and return path to annotated image"""
        if not coordinates_with_text:
            return image_path
        
        try:
            # Open the image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            
            # Generate colors for each bounding box
            colors = self.generate_color_palette(len(coordinates_with_text))
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            # Draw each bounding box
            for i, (coords, text) in enumerate(coordinates_with_text):
                x1, y1, x2, y2 = coords
                color = colors[i]
                
                # Draw bounding box rectangle
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # Draw text label with background
                if font:
                    bbox = draw.textbbox((0, 0), f"{i+1}", font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    text_width, text_height = 10, 10
                
                # Draw background rectangle for text
                label_bg = [x1, y1-text_height-4, x1+text_width+8, y1]
                draw.rectangle(label_bg, fill=color)
                
                # Draw text
                if font:
                    draw.text((x1+4, y1-text_height-2), f"{i+1}", fill='white', font=font)
                else:
                    draw.text((x1+4, y1-text_height-2), f"{i+1}", fill='white')
            
            # Save annotated image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            annotated_path = f"tmp_rovodev_annotated_{base_name}_{random.randint(1000,9999)}.png"
            image.save(annotated_path)
            
            return annotated_path
            
        except Exception as e:
            print(f"Error drawing bounding boxes: {e}")
            return image_path
    
    def process_images(self, images: List, prompt: str, output_format: List[str], progress=gr.Progress()) -> Tuple[str, str]:
        """Process multiple images for OCR recognition"""
        print(f"=== Starting image processing ===")
        print(f"Number of images: {len(images) if images else 0}")
        print(f"Prompt: {prompt}")
        
        if not images:
            return i18n.get('please_upload'), ""
            
        if not self.model_loaded:
            try:
                progress(0.1, desc=i18n.get('loading_model'))
                self.load_model()
            except Exception as e:
                error_msg = i18n.get('model_load_failed', error=str(e))
                print(error_msg)
                return error_msg, ""
        
        results = []
        total_images = len(images)
        
        for idx, image in enumerate(images):
            try:
                print(f"\n--- Processing image {idx + 1}/{total_images} ---")
                progress((idx + 1) / total_images, 
                        desc=i18n.get('processing_image', current=idx + 1, total=total_images))
                
                # Check image type
                print(f"Image type: {type(image)}")
                print(f"Image value: {image}")
                
                # Handle Gradio file objects
                if hasattr(image, 'name'):
                    # Gradio File object, use its path directly
                    temp_image_path = image.name
                    cleanup_temp = False
                    print(f"Using Gradio file path: {temp_image_path}")
                else:
                    # Create temporary file for other cases
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        # Save PIL Image object to file
                        if hasattr(image, 'save'):
                            image.save(temp_file.name)
                            print(f"Saved PIL image to: {temp_file.name}")
                        else:
                            # Copy file if it's a file path
                            import shutil
                            shutil.copy(image, temp_file.name)
                            print(f"Copied file to: {temp_file.name}")
                        
                        temp_image_path = temp_file.name
                        cleanup_temp = True
                
                # Check if file exists
                if not os.path.exists(temp_image_path):
                    raise Exception(i18n.get('file_not_exist', path=temp_image_path))
                
                print(f"Image file size: {os.path.getsize(temp_image_path)} bytes")
                
                # Execute OCR recognition
                formatted_prompt = f"<image>\n{prompt}"
                print(f"Formatted prompt: {formatted_prompt}")
                
                print("Starting OCR inference...")
                
                # Capture standard output to get streaming results
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()
                
                try:
                    result = self.model.infer(
                        self.tokenizer,
                        prompt=formatted_prompt,
                        image_file=temp_image_path,
                        output_path='./',
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        save_results=False,
                        test_compress=True
                    )
                finally:
                    # Restore standard output
                    sys.stdout = old_stdout
                
                # Get captured output
                captured_text = captured_output.getvalue()
                
                # Use captured output if direct result is empty
                raw_result = ""
                if (result is None or str(result).strip() == "") and captured_text.strip():
                    raw_result = captured_text.strip()
                    print(f"Using captured output: {raw_result[:200]}...")
                elif result:
                    raw_result = str(result)
                    print(f"Using direct result: {raw_result[:200]}...")
                else:
                    print(f"Captured output: {captured_text[:200]}...")
                    if captured_text.strip():
                        raw_result = captured_text.strip()
                
                # Clean and format result
                cleaned_result = self.clean_ocr_result(raw_result)
                
                # Validate cleaned result
                if not cleaned_result or cleaned_result.strip() == "":
                    if raw_result:
                        # Use raw result if cleaned result is empty but raw result is not
                        cleaned_result = raw_result
                        print(i18n.get('empty_after_clean'))
                    else:
                        cleaned_result = i18n.get('empty_result')
                
                # Extract coordinates from raw result for annotation
                coordinates_with_text = self.extract_coordinates_from_text(raw_result)
                
                # Generate annotated image if coordinates found and image format is requested
                annotated_image_path = None
                if coordinates_with_text and isinstance(output_format, list) and 'image' in output_format:
                    annotated_image_path = self.draw_bounding_boxes(temp_image_path, coordinates_with_text)
                    print(f"Generated annotated image: {annotated_image_path}")
                
                # Print debug information
                print(i18n.get('raw_result_length', length=len(raw_result)))
                print(i18n.get('cleaned_result', 
                              result=cleaned_result[:200] + "..." if len(cleaned_result) > 200 else cleaned_result))
                if coordinates_with_text:
                    print(f"Found {len(coordinates_with_text)} coordinate regions")
                
                results.append({
                    'image_index': idx + 1,
                    'result': str(cleaned_result),
                    'status': 'success',
                    'coordinates': coordinates_with_text,
                    'original_image_path': temp_image_path,
                    'annotated_image_path': annotated_image_path
                })
                
                # Clean up temporary files (only if we created them)
                if cleanup_temp:
                    try:
                        os.unlink(temp_image_path)
                        print(i18n.get('cleanup_temp', path=temp_image_path))
                    except:
                        pass
                
            except Exception as e:
                error_msg = i18n.get('recognition_failed', error=str(e))
                print(f"Error: {error_msg}")
                traceback.print_exc()
                
                results.append({
                    'image_index': idx + 1,
                    'result': error_msg,
                    'status': 'error',
                    'coordinates': [],
                    'original_image_path': None,
                    'annotated_image_path': None
                })
        
        print(f"\n=== Processing complete, total results: {len(results)} ===")
        
        # Format results for display
        formatted_results = self.format_results(results, output_format)
        summary = self.generate_summary(results)
        
        print(f"Formatted results length: {len(formatted_results)}")
        print(f"Summary length: {len(summary)}")
        
        return formatted_results, summary
    
    def format_results(self, results: List[dict], output_format: List[str]) -> str:
        """Format recognition results for display"""
        formatted = i18n.get('results_title')
        
        # Handle case where output_format is still a string (backward compatibility)
        if isinstance(output_format, str):
            output_format = [output_format]
        
        for result in results:
            status_icon = "✅" if result['status'] == 'success' else "❌"
            formatted += f"## {i18n.get('results_tab')} {result['image_index']} {status_icon}\n\n"
            
            # Display annotated image if available and image format is selected
            if 'image' in output_format and result.get('annotated_image_path'):
                try:
                    # Convert image to base64 for display
                    with open(result['annotated_image_path'], 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                    formatted += f"### Annotated Image with Bounding Boxes\n"
                    formatted += f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%; height: auto;" />\n\n'
                    
                    # Display coordinate legend if coordinates exist
                    if result.get('coordinates'):
                        formatted += "### Coordinate Legend\n"
                        colors = self.generate_color_palette(len(result['coordinates']))
                        for i, (coords, text) in enumerate(result['coordinates']):
                            color = colors[i]
                            formatted += f"<span style='color: {color}; font-weight: bold;'>{i+1}.</span> "
                            formatted += f"**[{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}]** - {text}\n\n"
                        formatted += "\n"
                except Exception as e:
                    print(f"Error displaying annotated image: {e}")
            
            # Handle result content
            result_content = result['result']
            if not result_content or str(result_content).strip() == "":
                result_content = i18n.get('no_result')
            
            # Display content in requested formats
            for fmt in output_format:
                if fmt == 'image':
                    continue  # Already handled above
                
                formatted += f"### {fmt.upper()} Format\n"
                
                if fmt.lower() == "html":
                    # HTML format - display content as HTML
                    if len(str(result_content)) > 1000:
                        formatted += f"<details>\n<summary>View full result (Length: {len(str(result_content))} characters)</summary>\n\n"
                        formatted += f"<div style='border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;'>\n{result_content}\n</div>\n\n"
                        formatted += "</details>\n\n"
                    else:
                        formatted += f"<div style='border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;'>\n{result_content}\n</div>\n\n"
                else:
                    # Markdown format - display content in code blocks
                    if len(str(result_content)) > 1000:
                        formatted += f"<details>\n<summary>View full result (Length: {len(str(result_content))} characters)</summary>\n\n"
                        formatted += f"```\n{result_content}\n```\n\n"
                        formatted += "</details>\n\n"
                    else:
                        formatted += f"```\n{result_content}\n```\n\n"
            
            formatted += "---\n\n"
        
        return formatted
    
    def generate_summary(self, results: List[dict]) -> str:
        """Generate recognition summary"""
        total = len(results)
        success = sum(1 for r in results if r['status'] == 'success')
        failed = total - success
        
        summary = i18n.get('summary_title')
        summary += f"- {i18n.get('total_images')}: {total}\n"
        summary += f"- {i18n.get('successful')}: {success}\n"
        summary += f"- {i18n.get('failed')}: {failed}\n"
        summary += f"- {i18n.get('success_rate')}: {(success/total*100):.1f}%\n\n"
        
        return summary
    
    def cleanup_temp_files(self, results: List[dict]):
        """Clean up temporary annotated image files"""
        for result in results:
            if result.get('annotated_image_path') and os.path.exists(result['annotated_image_path']):
                try:
                    os.unlink(result['annotated_image_path'])
                    print(f"Cleaned up temporary file: {result['annotated_image_path']}")
                except Exception as e:
                    print(f"Error cleaning up {result['annotated_image_path']}: {e}")

# Create OCR application instance
ocr_app = OCRApp()

# Wrapper function for processing with cleanup
def process_images_with_cleanup(images, prompt, output_format):
    """Process images and schedule cleanup of temporary files"""
    results_text, summary = ocr_app.process_images(images, prompt, output_format)
    
    # Schedule cleanup of temporary files after a delay
    import threading
    import time
    
    def delayed_cleanup():
        time.sleep(30)  # Wait 30 seconds to allow UI to load images
        # Find all temp files created by this session
        import glob
        temp_files = glob.glob("tmp_rovodev_annotated_*.png")
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                print(f"Error cleaning up {temp_file}: {e}")
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=delayed_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    return results_text, summary

# Language change handler
def change_language(language):
    """Handle language change"""
    i18n.set_language(language)
    print(f"Language changed to: {language}")
    return create_interface_components()

# Preset prompt handlers
def set_preset_general():
    return "Free OCR."

def set_preset_markdown():
    return "<|grounding|>Convert the document to markdown."

def set_preset_table():
    return "<|grounding|>Extract all tables and convert to markdown format."

def create_interface_components():
    """Create interface components with current language"""
    return (
        # Update all component labels and values
        gr.update(label=i18n.get('upload_label')),
        gr.update(label=i18n.get('prompt_label'), 
                 placeholder=i18n.get('prompt_placeholder')),
        gr.update(label="Output Format / 输出格式", 
                 info="Choose how to display the OCR results / 选择OCR结果的显示方式"),
        gr.update(value=i18n.get('preset_general')),
        gr.update(value=i18n.get('preset_markdown')),
        gr.update(value=i18n.get('preset_table')),
        gr.update(value=i18n.get('recognize_btn')),
        gr.update(label=i18n.get('results_tab')),
        gr.update(label=i18n.get('summary_tab')),
        gr.update(value=i18n.get('waiting_message')),
        gr.update(value=i18n.get('waiting_summary')),
        gr.update(value=i18n.get('instructions_content'))
    )

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface"""
    with gr.Blocks(title=i18n.get('app_title'), theme=gr.themes.Soft()) as demo:
        # Header with title, GitHub link and language switcher
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(f"# {i18n.get('app_title')}")
            with gr.Column(scale=1, min_width=120):
                github_btn = gr.HTML("""
                    <div style="text-align: center; margin-top: 10px;">
                        <a href="https://github.com/newlxj/DeepSeek-OCR-Web-UI" target="_blank" style="
                            display: inline-flex;
                            align-items: center;
                            padding: 8px 16px;
                            background-color: #f6f8fa;
                            color: #24292e;
                            text-decoration: none;
                            border: 1px solid #d1d9e0;
                            border-radius: 6px;
                            font-size: 14px;
                            font-weight: 500;
                            transition: all 0.2s;
                        " onmouseover="this.style.backgroundColor='#f3f4f6'; this.style.borderColor='#c9d1d9';" onmouseout="this.style.backgroundColor='#f6f8fa'; this.style.borderColor='#d1d9e0';">
                            <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16" style="margin-right: 6px;">
                                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                            </svg>
                            GitHub
                        </a>
                    </div>
                """)
            with gr.Column(scale=1, min_width=150):
                language_selector = gr.Dropdown(
                    choices=[('English', 'en'), ('中文', 'zh')],
                    value='en',
                    label=i18n.get('language_label'),
                    interactive=True
                )
        
        gr.Markdown(i18n.get('app_description'))
        
        with gr.Row():
            with gr.Column(scale=1):
                # Image upload area
                images_input = gr.File(
                    label=i18n.get('upload_label'),
                    file_count="multiple",
                    file_types=["image"],
                    height=200
                )
                
                # Prompt input
                prompt_input = gr.Textbox(
                    label=i18n.get('prompt_label'),
                    value="<|grounding|>Convert the document to markdown.",
                    placeholder=i18n.get('prompt_placeholder'),
                    lines=3
                )
                
                # Output format selector (multi-select)
                output_format = gr.CheckboxGroup(
                    choices=[("Markdown", "markdown"), ("HTML", "html"), ("Annotated Image", "image")],
                    value=["markdown"],
                    label="Output Format / 输出格式",
                    info="Choose how to display the OCR results (multiple selections allowed) / 选择OCR结果的显示方式（可多选）"
                )
                
                # Preset prompt buttons
                with gr.Row():
                    preset_ocr = gr.Button(i18n.get('preset_general'), size="sm")
                    preset_markdown = gr.Button(i18n.get('preset_markdown'), size="sm")
                    preset_table = gr.Button(i18n.get('preset_table'), size="sm")
                
                # Recognition button
                recognize_btn = gr.Button(
                    i18n.get('recognize_btn'), 
                    variant="primary", 
                    size="lg"
                )
            
            with gr.Column(scale=2):
                # Results display area
                with gr.Tab(i18n.get('results_tab')):
                    results_output = gr.Markdown(
                        label=i18n.get('results_tab'),
                        value=i18n.get('waiting_message'),
                        height=500
                    )
                
                with gr.Tab(i18n.get('summary_tab')):
                    summary_output = gr.Markdown(
                        label=i18n.get('summary_tab'),
                        value=i18n.get('waiting_summary'),
                        height=500
                    )
        
        # Instructions section
        instructions_md = gr.Markdown(i18n.get('instructions_content'))
        
        # Language change event
        language_selector.change(
            fn=lambda lang: change_language(lang),
            inputs=[language_selector],
            outputs=[
                images_input, prompt_input, output_format, preset_ocr, preset_markdown, 
                preset_table, recognize_btn, results_output, summary_output,
                results_output, summary_output, instructions_md
            ]
        )
        
        # Preset prompt click events
        preset_ocr.click(fn=set_preset_general, outputs=prompt_input)
        preset_markdown.click(fn=set_preset_markdown, outputs=prompt_input)
        preset_table.click(fn=set_preset_table, outputs=prompt_input)
        
        # Recognition button click event
        recognize_btn.click(
            fn=process_images_with_cleanup,
            inputs=[images_input, prompt_input, output_format],
            outputs=[results_output, summary_output],
            show_progress=True
        )
    
    return demo

# Launch application
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        # server_name="0.0.0.0",
        server_name="localhost",
        server_port=7860,
        share=True,
        show_error=True
    )
