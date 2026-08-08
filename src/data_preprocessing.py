import re
import unicodedata
import pandas as pd


# ──────────────────────────────────────────────
# Dialect column mapping
# (matches the Excel column names exactly)
# ──────────────────────────────────────────────
DIALECT_COLUMNS = {
    "standard":    "'শুদ্ধ বাংলা ভাষা'",
    "rajshahi":    "'রাজশাহী'",
    "sylheti":     "'সিলেট'",
    "chittagong":  "'চিটাগং'",
    "rangpur":     "'রংপুর'",
    "mymensingh":  "'ময়মনসিংহ'",
    "barishal":    "'বরিশাল'",
    "rakhain":     "'রাখাইন'",
}


# ──────────────────────────────────────────────
# Text cleaning helpers
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Basic Bangla text cleaning:
    - NFC unicode normalization
    - Remove non-Bangla characters (keep Bangla Unicode block + whitespace)
    - Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\u0980-\u09FF\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_bangla(text: str) -> str:
    """
    Normalize common Unicode variations that appear
    as different codepoints but look identical.
    Add more pairs here as you discover them in your data.
    """
    if not isinstance(text, str):
        return ""

    replacements = {
        "\u09CB": "\u09CB",  # ো  (kept as reference — extend this list)
        "\u09CB": "\u09CB",  # ে  (extend as needed)
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def preprocess_pipeline(text: str) -> str:
    """Full preprocessing: clean → normalize."""
    text = clean_text(text)
    text = normalize_bangla(text)
    return text


# ──────────────────────────────────────────────
# Excel loading  ← NEW
# ──────────────────────────────────────────────

def load_excel_dataset(excel_path: str) -> pd.DataFrame:
    """
    Load the dialect Excel file and return a clean DataFrame.

    Output columns (English keys for easy use in code):
        standard, rajshahi, sylheti, chittagong,
        rangpur, mymensingh, barishal, rakhain

    Each cell is preprocessed with preprocess_pipeline().
    Rows where the standard column is empty are dropped.
    """
    df_raw = pd.read_excel(excel_path)

    # Build a renamed DataFrame using positional columns
    # (the Excel headers contain Bangla text with quotes)
    col_positions = list(DIALECT_COLUMNS.keys())   # English names
    col_indices   = list(range(len(col_positions))) # 0-7

    df = pd.DataFrame()
    for eng_name, idx in zip(col_positions, col_indices):
        df[eng_name] = df_raw.iloc[:, idx].astype(str).str.strip()

    # Drop rows where 'standard' is empty or NaN placeholder
    df = df[df["standard"].notna()]
    df = df[df["standard"] != ""]
    df = df[df["standard"] != "nan"]
    df = df.reset_index(drop=True)

    # Apply preprocessing to every cell
    for col in col_positions:
        df[col] = df[col].apply(preprocess_pipeline)

    # Drop rows that became empty after cleaning
    df = df[df["standard"] != ""].reset_index(drop=True)

    return df


def flatten_to_samples(df: pd.DataFrame) -> list[dict]:
    """
    Convert the parallel DataFrame into a flat list of samples.

    Each sample:
        {
            "text":     <dialect sentence>,
            "standard": <standard Bangla equivalent>,
            "dialect":  <dialect name, e.g. "sylheti">
        }

    This format feeds directly into BanglaDialectDataset.
    The standard column itself is also included as a sample
    (labelled "standard") so the model sees it too.
    """
    dialect_names = [
        "standard", "rajshahi", "sylheti", "chittagong",
        "rangpur",  "mymensingh", "barishal", "rakhain",
    ]

    samples = []
    for _, row in df.iterrows():
        standard_text = row["standard"]
        for dialect in dialect_names:
            text = row[dialect]
            if text and text != "nan":
                samples.append({
                    "text":     text,
                    "standard": standard_text,
                    "dialect":  dialect,
                })
    return samples


# ──────────────────────────────────────────────
# Legacy plain-text helpers (kept for compatibility)
# ──────────────────────────────────────────────

def preprocess_file(input_path: str, output_path: str) -> None:
    """Preprocess a plain .txt file line by line."""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    processed = [preprocess_pipeline(line) for line in lines if line.strip()]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(processed))

    print(f"Processed {len(processed)} lines → {output_path}")