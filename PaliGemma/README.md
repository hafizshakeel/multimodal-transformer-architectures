# PaliGemma: Vision-Language Model Implementation

This repository contains an educational implementation of Google's PaliGemma, a powerful multimodal model that combines SigLIP vision encoder with Gemma language model for vision-language tasks.


## Installation

1. Clone this repository:
```bash
git clone https://github.com/hafizshakeel/multimodal-transformer-architectures
cd paligemma-pytorch/
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the model weights:
```bash
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

## Usage

### Running Inference

You can use the provided script to run inference:

```bash
bash launch_inference.sh
```

Or run the Python script directly:

```bash
python inference.py \
    --model_path "path/to/paligemma-3b-pt-224" \
    --prompt "Describe this image:" \
    --image_file_path "path/to/image.jpg" \
    --max_tokens_to_generate 100 \
    --temperature 0.8 \
    --top_p 0.9 \
    --do_sample True \
    --only_cpu False
```

### Parameters

- `model_path`: Path to the model directory
- `prompt`: Text prompt to condition the generation
- `image_file_path`: Path to the input image
- `max_tokens_to_generate`: Maximum number of tokens to generate
- `temperature`: Controls randomness (higher = more random)
- `top_p`: Controls diversity via nucleus sampling
- `do_sample`: Whether to sample or use greedy decoding
- `only_cpu`: Whether to use CPU instead of GPU

## Key Components

1. **PaliGemma Architecture** (`paliGemma.py`):
   - Implementation of the multimodal model combining vision and language
   - Includes KV-Cache for efficient generation
   - Implements Rotary Position Embeddings (RoPE)
   - Implements Grouped Query Attention (GQA)

2. **SigLIP Vision Encoder** (`siglip.py`):
   - Vision transformer for encoding images
   - Detailed implementation of attention mechanisms

3. **Preprocessing Pipeline** (`preprocessing.py`):
   - Image processing functions
   - Text tokenization and formatting
   - Handling of special tokens for multimodal inputs

4. **Inference Engine** (`inference.py`):
   - Autoregressive generation implementation
   - Top-p (nucleus) sampling
   - Efficient token generation with KV caching


## License

This project is released under the MIT License.

## Acknowledgments

This implementation is based on the architecture described in the [PaliGemma paper](https://arxiv.org/abs/2407.07726) by Google Research, and the unofficial implementation by [hkproj](https://github.com/hkproj/pytorch-paligemma). The original model can be found at [Google/PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224).
