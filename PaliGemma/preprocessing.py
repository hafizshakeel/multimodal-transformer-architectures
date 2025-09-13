from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

"""
Image and text preprocessing utilities for PaliGemma.

This module handles the processing of images and text inputs for the PaliGemma model,
including resizing, normalization, tokenization, and proper formatting for model input.
"""

# Standard ImageNet normalization values - PaliGemma uses these values for image normalization
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]  # Mean per RGB channel
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]   # Standard deviation per RGB channel

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    """
    Format a text prompt with image tokens as required by PaliGemma.
    
    This function prepends image tokens and appends appropriate tokens to text prompts
    following PaliGemma's expected input format.
    
    Args:
        prefix_prompt: The user's text prompt
        bos_token: Beginning of sequence token
        image_seq_len: Number of image tokens to add
        image_token: The special token string used for image representation
        
    Returns:
        Formatted string with image tokens + BOS + prompt + newline
    """
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n" # (image + bos + prompt + \n)


def resize(image: Image, size: Tuple[int, int], resample: Image.Resampling = None, reducing_gap: Optional[int] = None) -> np.ndarray:
    """
    Resize an image to the specified dimensions.
    
    Args:
        image: PIL Image to resize
        size: Target (height, width) size
        resample: PIL resampling filter to use
        reducing_gap: Optional parameter for PIL's resize
        
    Returns:
        Resized image
    """
    height, width = size
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)
    return resized_image


def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Rescale image pixel values by multiplying by a scale factor.
    
    Typically used to convert uint8 images [0-255] to float [0-1] range.
    
    Args:
        image: Image as numpy array
        scale: Scale factor to apply (e.g., 1/255.0)
        dtype: Output data type
        
    Returns:
        Rescaled image array
    """
    rescale_image = image * scale
    rescale_image = rescale_image.astype(dtype)
    return rescale_image


def normalize(image: np.ndarray, mean: Union[float, Iterable[float]], std: Union[float, Iterable[float]]) -> np.ndarray:
    """
    Apply mean and standard deviation normalization to an image.
    
    Implements the standard (image - mean) / std normalization formula.
    
    Args:
        image: Input image array
        mean: Mean value(s) to subtract
        std: Standard deviation value(s) to divide by
        
    Returns:
        Normalized image array
    """
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_images(
        images: List[Image.Image],
        size: Dict[str, int] = None,
        resample: Image.Resampling = None,
        rescale_factor: float = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    Process a batch of images for input to the vision encoder.
    
    This function performs several preprocessing steps:
    1. Resize images to the target size
    2. Convert PIL images to numpy arrays
    3. Rescale pixel values (typically to [0,1] range)
    4. Apply normalization
    5. Transpose from HWC to CHW format (PyTorch expects channels first)
    
    Args:
        images: List of PIL Images to process
        size: Target size as (height, width)
        resample: PIL resampling filter to use
        rescale_factor: Factor to multiply pixel values by
        image_mean: Mean values for normalization
        image_std: Standard deviation values for normalization
        
    Returns:
        List of processed image arrays ready for model input
    """
    height, width = size[0], size[1]
    
    images = [resize(image=image, size=(height, width), resample=resample) for image in images]
    images = [np.array(image) for image in images]
    images = [rescale(image, scale=rescale_factor) for image in images]
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    images = [image.transpose(2, 0, 1) for image in images]
    
    return images


class PaliGemmaProcessor:
    """
    Processor class that handles both image and text inputs for PaliGemma.
    
    This class prepares inputs by:
    1. Processing images to the right format
    2. Tokenizing text with appropriate tokens
    3. Combining them in the format expected by the model
    
    The main entry point is the __call__ method, which handles the full processing pipeline.
    """
    
    # Special token used as a placeholder for image embeddings
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        """
        Initialize the PaliGemma processor.
        
        Args:
            tokenizer: Hugging Face tokenizer
            num_image_tokens: Number of image tokens to use (matches the patch grid size)
            image_size: Size of images (both height and width)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Add special tokens to the tokenizer
        # Study Tokenizer -> https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        
        # Add additional tokens used for various vision tasks
        # Location tokens used for object detection
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  
        
        # Segmentation tokens used for object segmentation
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  
        
        tokenizer.add_tokens(EXTRA_TOKENS)
        
        # Get the token ID for the image token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        
        # Control token addition behavior
        # We will add the beginning-of-sentence (BOS) and end-of-sentence (EOS) tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",  # The input is padded to the length of the longest sequence
        truncation: bool = True,
    ) -> dict:
        """
        Process text and images for model input.
        
        Args:
            text: List of text prompts
            images: List of PIL Images
            padding: Padding strategy for tokenization
            truncation: Whether to truncate sequences that are too long
            
        Returns:
            Dictionary containing:
                - pixel_values: Processed image tensor
                - input_ids: Token IDs including image placeholders
                - attention_mask: Attention mask for the sequence
        """
        # Currently only supporting batch size of 1
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        # Process the images with standard PaliGemma image processing pipeline
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,  # High-quality resampling
            rescale_factor=1 / 255.0,           # Convert from [0-255] to [0-1]
            image_mean=IMAGENET_STANDARD_MEAN,  # Standard normalization
            image_std=IMAGENET_STANDARD_STD,
        )

        # Convert the list of numpy arrays to a single numpy array of shape (B, C, H, W)
        # B = batch size, C = channels, H = height, W = width
        pixel_values = np.stack(pixel_values, axis=0)
        
        # Convert the numpy array to a PyTorch tensor
        pixel_values = torch.tensor(pixel_values)

        # Format each prompt with the appropriate tokens
        # Prepend image tokens and add BOS token and newline
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Tokenize the formatted prompts to get input_ids and attention_mask
        inputs = self.tokenizer(
            input_strings,
            return_tensors='pt',  # Return PyTorch tensors
            padding=padding,      # Pad sequences to the same length
            truncation=truncation,
        )

        # Combine the processed images and tokenized text in one dictionary
        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data


# Example usage code (commented out)
# from transformers import AutoTokenizer
#
# # Assume PaliGemmaProcessor and helper functions are defined above.
#
# # Initialize tokenizer (GPT2Tokenizer as an example)
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have a pad token by default
#
# # Initialize processor
# processor = PaliGemmaProcessor(
#     tokenizer=tokenizer,
#     num_image_tokens=4,      # number of <image> tokens to insert
#     image_size=224           # standard image size
# )
#
# # Create dummy image (RGB)
# dummy_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
#
# # Input prompt
# sample_text = ["Describe the contents of the image."]
#
# # Run the processor
# output = processor(
#     text=sample_text,
#     images=[dummy_image]
# )
#
# # output = processor(
# #     text=["Short.", "A very long sentence that describes the contents of a complex image in great detail."],
# #     images=[dummy_image, dummy_image]
# # )
# #
# # print(output["attention_mask"])
#
#
# # Extract parts
# input_ids = output["input_ids"]
# attention_mask = output["attention_mask"]
# pixel_values = output["pixel_values"]
#
# # Decode the tokens back to string for inspection
# decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
#
# # Print everything
# print("\n====== STRING VERSION (decoded input) ======\n")
# print(decoded_prompt)
#
# print("\n====== TOKEN IDS (input_ids) ======\n")
# print(input_ids)
#
# print("\n====== ATTENTION MASK ======\n")
# print(attention_mask)
#
# print("\n====== PIXEL VALUES SHAPE ======")
# print(pixel_values.shape)

