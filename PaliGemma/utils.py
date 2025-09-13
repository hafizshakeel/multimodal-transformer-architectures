from paliGemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

"""
Utility functions for loading and handling PaliGemma models.
This module provides helpers for loading model weights and configurations.
"""

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    Loads a PaliGemma model and tokenizer from HuggingFace-style saved weights.
    
    This function:
    1. Loads the tokenizer from the specified path
    2. Finds and loads weights from safetensors files
    3. Creates the model with the correct configuration
    4. Initializes the model with the loaded weights
    
    Args:
        model_path: Path to the directory containing model files
        device: Device to load the model onto ('cpu', 'cuda', etc.)
        
    Returns:
        Tuple containing:
            - Initialized PaliGemma model
            - Tokenizer for text processing
    """
    # Load the tokenizer with padding on the right side (important for generation)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right", "Tokenizer must use right padding for proper generation"

    # Find all safetensors weight files (weights are split across multiple files)
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # Load weights from each file into a unified dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's configuration from the config file
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model instance using the loaded configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Initialize the model with the loaded weights
    # strict=False allows loading even if some keys don't match exactly
    model.load_state_dict(tensors, strict=False)

    # Tie the weights between embedding layer and output layer
    # This is a common technique in language models to share parameters
    model.tie_weights()

    return (model, tokenizer)