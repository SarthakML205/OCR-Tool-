# Image Text Extraction with GOT-OCR

This Streamlit application allows users to upload images and extract text from them using the GOT (General OCR Theory) model. Additionally, users can search for specific words or phrases within the extracted text, and the app will highlight them.

## Features

- **Image Upload**: Upload images in JPG, PNG, or JPEG formats.
- **Text Extraction**: Uses the GOT model to extract text from uploaded images.
- **Search Functionality**: Enter a word or phrase to search in the extracted text, and it will be highlighted in the output.
- **Text Export**: Save the extracted text in JSON format for later use.

## Demo

Try the live demo on Hugging Face: [GOT-OCR](https://justin4602-ocr.hf.space)

## Setup

### Prerequisites

- Python 3.7 or above
- Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

1. Clone this repository.
2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## How It Works

### 1. Model Initialization

The app uses the `GOT` model for Optical Character Recognition (OCR). The model is loaded using the Hugging Face model hub.

**Model Initialization Function:**

```python
@st.cache_resource
def init_model():
    tokenizer = AutoTokenizer.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True)
    model = AutoModel.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
    model = model.eval()  # Set the model to evaluation mode
    return model, tokenizer
```

For environments that support GPU, there's an option to initialize the model using the GPU for faster performance:

```python
def init_gpu_model():
    tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
    model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
    model = model.eval().cuda()
    return model, tokenizer
```

### 2. Text Extraction

Once an image is uploaded, the app uses the model to extract text.

```python
@st.cache_data
def get_text(image_file, _model, _tokenizer):
    res = _model.chat(_tokenizer, image_file, ocr_type='ocr')
    return res
```

### 3. Search and Highlight

Users can input a search term, and the app will highlight all occurrences in the extracted text.

```python
def highlight_text(text, search_term):
    if not search_term:
        return text
    pattern = re.compile(re.escape(search_term), re.IGNORECASE)
    return pattern.sub(lambda m: f'<span style="background-color: grey;">{m.group()}</span>', text)
```

### 4. Save Extracted Text

The extracted text can be saved to a JSON file.

```python
def save_text_to_json(file_name, text_data):
    """Save the extracted text into a JSON file."""
    with open(file_name, 'w') as json_file:
        json.dump({"extracted_text": text_data}, json_file, indent=4)
    st.success(f"Text saved to {file_name}")
```

## Notes

- The app uses the **CPU version** of the GOT model, which ensures compatibility across environments without GPU support. However, it may be slower than a GPU-enabled model.
- The search functionality is **case-insensitive** and highlights all matches in the extracted text.

## Directory Structure

The app temporarily stores uploaded images in the `images` directory. This directory is created automatically when the app runs.

## Performance Considerations

The CPU version of the GOT model is used for wider compatibility, but if you're running the app on a machine with GPU support, you can modify the code to use the GPU model for faster performance.

## Contributing

Feel free to contribute by submitting a pull request or suggesting features.

## License

This project is licensed under the MIT License.

---

You can update the details (e.g., dependencies) as per your project's specific requirements. Let me know if you need any further modifications!
