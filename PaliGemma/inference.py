from PIL import Image
import torch
import fire

from preprocessing import PaliGemmaProcessor
from paliGemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

"""
Inference script for PaliGemma models.

This module provides functions for running text-to-image generation with PaliGemma,
including handling device placement, input processing, and autoregressive generation.
"""

def move_inputs_to_device(model_inputs: dict, device: str) -> dict:
    """
    Move all tensors in the input dictionary to the specified device.
    
    Args:
        model_inputs: Dictionary of input tensors
        device: Target device ('cpu', 'cuda', 'mps')
        
    Returns:
        Dictionary with all tensors moved to the target device
    """
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, 
    prompt: str, 
    image_file_path: str, 
    device: str
) -> dict:
    """
    Prepare inputs for the PaliGemma model from a text prompt and image file.
    
    This function:
    1. Loads the image from disk
    2. Processes both image and text through the processor
    3. Moves all inputs to the specified device
    
    Args:
        processor: PaliGemma processor for handling inputs
        prompt: Text prompt to condition the generation
        image_file_path: Path to the image file
        device: Target device for tensors
        
    Returns:
        Dictionary containing processed model inputs
    """
    # Load and prepare the image
    image = Image.open(image_file_path)
    images = [image]  # Processor expects a list of images
    prompts = [prompt]  # Processor expects a list of prompts
    
    # Process inputs through the PaliGemma processor
    model_inputs = processor(text=prompts, images=images)
    
    # Move all tensors to the target device
    model_inputs = move_inputs_to_device(model_inputs, device)
    
    return model_inputs


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    """
    Run inference with PaliGemma to generate text based on an image and prompt.
    
    This implements autoregressive generation:
    1. Process the initial inputs (image + prompt)
    2. Generate tokens one by one until max length or stop token
    3. For each step, use previous tokens and KV cache for efficiency
    4. Sample from the output distribution using temperature and top-p sampling
    
    Args:
        model: PaliGemma model instance
        processor: Processor for handling inputs
        device: Target device for tensors
        prompt: Text prompt to condition generation
        image_file_path: Path to the input image
        max_tokens_to_generate: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random)
        top_p: Controls diversity via nucleus sampling
        do_sample: Whether to sample or use greedy decoding
        
    Returns:
        None (prints the generated text)
    """
    # Get processed inputs for the model
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    # Initialize the key-value cache for efficient generation
    kv_cache = KVCache()

    # Keep track of generated tokens
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    # Generate tokens autoregressively (one at a time)
    for _ in range(max_tokens_to_generate):
        # Forward pass through the model
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        
        # Get the updated KV cache (avoids recomputing previous tokens)
        kv_cache = outputs["kv_cache"]
        
        # Get the logits for the next token (last position only)
        next_token_logits = outputs["logits"][:, -1, :]
        
        # Determine the next token based on sampling strategy
        if do_sample:
            # Apply temperature scaling to control randomness
            # Higher temperature = more random, lower = more deterministic
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            # Use nucleus (top-p) sampling
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            # Greedy decoding - just take the most likely token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
        # Ensure proper shape: [1, 1]
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # Remove batch dimension
        
        # Add token to our generated sequence
        generated_tokens.append(next_token)
        
        # Stop if we've generated the stop token
        if next_token.item() == stop_token:
            break
            
        # Prepare inputs for the next iteration
        # For the next step, we only need the newly generated token as input
        input_ids = next_token.unsqueeze(-1)
        
        # Extend the attention mask for the new token
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    # Combine all generated tokens into a single tensor
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    
    # Decode the generated tokens back to text
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Print the full response (prompt + generated text)
    print(prompt + decoded)


def _sample_top_p(probs: torch.Tensor, p: float):
    """
    Nucleus sampling (top-p) implementation for text generation.
    
    This sampling strategy:
    1. Sorts token probabilities in descending order
    2. Keeps only the top tokens whose cumulative probability exceeds p
    3. Samples from this reduced set of tokens
    
    Args:
        probs: Token probability distribution
        p: Probability threshold (e.g., 0.9 = top 90% probability mass)
        
    Returns:
        Tensor containing the sampled token ID
    """
    # Sort probabilities in descending order and keep track of indices
    # Shape: (batch_size, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    
    # Calculate cumulative probabilities
    # Shape: (batch_size, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    
    # Create a mask for tokens that are beyond our probability threshold
    # Subtracting probs_sort shifts the cumulative sum by 1 position 
    # to check if the cumulative prob EXCLUDING current token exceeds p
    mask = probs_sum - probs_sort > p
    
    # Zero out all probabilities beyond the nucleus
    probs_sort[mask] = 0.0
    
    # Renormalize the remaining probabilities to sum to 1
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    
    # Sample a token index from the reduced distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    
    # Convert the sampled index back to the actual token ID
    next_token = torch.gather(probs_idx, -1, next_token)
    
    return next_token


def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    """
    Main entry point for PaliGemma inference.
    
    Handles:
    1. Device selection and setup
    2. Model and processor loading
    3. Running the generation process
    
    Args:
        model_path: Path to the PaliGemma model directory
        prompt: Text prompt to condition the generation
        image_file_path: Path to the input image
        max_tokens_to_generate: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random)
        top_p: Controls diversity via nucleus sampling
        do_sample: Whether to sample or use greedy decoding
        only_cpu: Force CPU usage even if GPU is available
    """
    # Determine which device to use (CUDA, MPS, or CPU)
    device = "cpu"

    if not only_cpu:
        # Try to use GPU if available and not explicitly disabled
        if torch.cuda.is_available():
            device = "cuda"  # NVIDIA GPU
        elif torch.backends.mps.is_available():
            device = "mps"   # Apple Silicon GPU

    print("Device in use: ", device)

    # Load the model and tokenizer
    print(f"Loading model")
    model, tokenizer = load_hf_model(model_path, device)
    
    # Ensure model is in evaluation mode (disables dropout, etc.)
    model = model.to(device).eval()

    # Create a processor for handling inputs
    # Extract parameters from the model configuration
    num_image_tokens = model.config.text_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    # Run the generation process
    print("Running inference")
    with torch.no_grad():  # Disable gradient tracking for inference
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


# Allow calling this script from the command line with arguments
if __name__ == "__main__":
    fire.Fire(main)
