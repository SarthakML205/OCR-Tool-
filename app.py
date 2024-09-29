from transformers import AutoModel, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
import streamlit as st
import os
from PIL import Image
import torch
from torchvision import io
import re
from typing import Dict

@st.cache_resource
def load_model():
    # Loading the CPU-based model for GOT OCR
    token = AutoTokenizer.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True)
    model = AutoModel.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True, use_safetensors=True, pad_token_id=token.eos_token_id)
    model.eval()
    return model, token

def load_gpu_model():
    # Initialize the GPU model for faster inference
    token = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
    model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=token.eos_token_id)
    model.eval().cuda()
    return model, token

def load_qwen_vl_model():
    # Load Qwen model for handling visual-linguistic tasks
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", device_map="cpu", torch_dtype=torch.float16)
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model, proc

def process_qwen_output(image_file, model, processor):
    try: 
        img = Image.open(image_file).convert('RGB')
        # Setting up the conversation format for Qwen2-VL
        dialogue = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Extract text from this image."}
                ]
            }
        ]
        text_query = processor.apply_chat_template(dialogue, add_generation_prompt=True)
        inputs = processor(text=[text_query], images=[img], padding=True, return_tensors="pt")
        inputs = {k: v.to(torch.float32) if torch.is_floating_point(v) else v for k, v in inputs.items()}

        config = {
            "max_new_tokens": 32,
            "do_sample": False,
            "top_k": 20,
            "top_p": 0.9,
            "temperature": 0.4,
            "num_return_sequences": 1,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }

        output_ids = model.generate(**inputs, **config)
        generated_ids = output_ids[:, inputs.get('input_ids', torch.empty(0)).shape[1]:]

        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0] if output_text else "No text found in the image."
    
    except Exception as e:
        return f"Error: {str(e)}"

@st.cache_data
def extract_text_from_image(image_path, model, tokenizer):
    # Extract text using the specified model and tokenizer
    result = model.chat(tokenizer, image_path, ocr_type='ocr')
    return result

def emphasize_text(text, term):
    # Highlight occurrences of search term in the extracted text
    if not term:
        return text
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    return pattern.sub(lambda match: f'<span style="background-color: yellow;">{match.group()}</span>', text)

st.title("OCR Tool - GOT OCR 2.0")
st.write("Upload an image to perform OCR and extract text.")

# Load the CPU model
OCR_MODEL, OCR_TOKENIZER = load_model()

image_file = st.file_uploader("Upload your image (JPG/PNG/JPEG)", type=['jpg', 'png', 'jpeg'])

if image_file:
    # Create folder for storing images if it doesn't exist
    if not os.path.exists("image_uploads"):
        os.makedirs("image_uploads")
    
    img_path = os.path.join("image_uploads", image_file.name)
    with open(img_path, "wb") as img_file:
        img_file.write(image_file.getbuffer())

    # Extract text from the uploaded image
    ocr_text = extract_text_from_image(img_path, OCR_MODEL, OCR_TOKENIZER)
    
    # Search functionality: Search a term in the extracted text
    search_input = st.text_input("Search for a term:")
    highlighted_result = emphasize_text(ocr_text, search_input)
    
    st.markdown(highlighted_result, unsafe_allow_html=True)
