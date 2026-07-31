from transformers import AutoTokenizer
import yaml

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_tokenizer(config_path="configs/config.yaml"):
    config = load_config(config_path)
    tokenizer_name = config["tokenizer"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Tokenizer loaded: {tokenizer_name}")
    return tokenizer