# Multimodal Transformer Architectures

This repository contains PyTorch implementations of various state-of-the-art multimodal transformer architectures. Each implementation is designed to be educational, focusing on clear code structure and detailed comments to help understand these complex models.

## Currently Implemented Models

- [PaliGemma](./PaliGemma/): A multimodal model combining SigLIP vision encoder with Gemma language model for vision-language tasks

## Repository Structure

Each subfolder contains a self-contained implementation of a specific architecture:

```
multimodal-transformer-architectures/
├── LICENSE
├── README.md
├── PaliGemma/
│   ├── inference.py          # Inference script for image-to-text generation
│   ├── launch_inference.sh   # Shell script to run inference
│   ├── paliGemma.py          # Core implementation of the PaliGemma model
│   ├── preprocessing.py      # Input preprocessing utilities
│   ├── README.md             # Detailed documentation for PaliGemma
│   ├── requirements.txt      # Dependencies for PaliGemma
│   ├── siglip.py             # SigLIP vision encoder implementation
│   └── utils.py              # Helper functions
└── requirements.txt          # Project-wide dependencies
```

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/hafizshakeel/multimodal-transformer-architectures.git
cd multimodal-transformer-architectures
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Running a Model

Navigate to the specific model's directory and follow the instructions in its README:

```bash
cd PaliGemma
# See the model-specific README for further instructions
```

## Model Details

### PaliGemma

PaliGemma is a vision-language model that combines:
- A SigLIP vision encoder for processing images
- A Gemma language model for text generation
- A multimodal projector to align vision and language representations

It can be used for tasks like image captioning, visual question answering, and image-conditioned text generation.

[Learn more about PaliGemma →](./PaliGemma/)

## Contributing

Contributions to add new multimodal architectures or improve existing implementations are welcome.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Email: hafizshakeel1997@gmail.com