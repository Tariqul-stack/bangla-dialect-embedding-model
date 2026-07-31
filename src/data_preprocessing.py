import re
import unicodedata

def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_bangla(text):
    # Normalize common dialect variations
    replacements = {
        'ো': 'ো',
        'ে': 'ে',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def preprocess_pipeline(text):
    text = clean_text(text)
    text = normalize_bangla(text)
    return text

def preprocess_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    processed = [preprocess_pipeline(line) for line in lines if line.strip()]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(processed))

    print(f"Processed {len(processed)} lines → {output_path}")