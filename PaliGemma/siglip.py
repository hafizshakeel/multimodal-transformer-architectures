import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
SigLIP Vision Encoder for PaliGemma

This module implements the SigLIP (Sigmoid Loss Prediction) vision encoder,
which is a variant of the CLIP architecture that uses a sigmoid-based
contrastive loss and improved training techniques.

SigLIP provides strong visual representations that serve as the vision
foundation for PaliGemma's multimodal capabilities.

Reference: https://arxiv.org/abs/2303.15343
"""

class SiglipVisionConfig:
    """
    Configuration class for the SigLIP vision encoder.
    
    This stores all hyperparameters required to build the vision transformer
    used as the image encoder in PaliGemma.
    """
    def __init__(
        self,
        hidden_size=1664,             # Dimension of the transformer
        intermediate_size=8192,       # Dimension of the MLP
        num_hidden_layers=32,         # Number of transformer layers  
        num_attention_heads=16,       # Number of attention heads
        image_size=224,               # Input image resolution
        patch_size=14,                # Size of image patches
        num_channels=3,               # Number of input channels (RGB)
        initializer_range=0.02,       # Initializer range
        initializer_factor=1.0,       # Initializer factor
        attention_dropout_rate=0.0,   # Dropout rate for attention
        dropout_rate=0.0,             # Dropout rate
        layernorm_eps=1e-6,           # Layer norm epsilon
        **kwargs
    ):
        """Initialize the SigLIP vision configuration with model hyperparameters."""
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout_rate = attention_dropout_rate
        self.dropout_rate = dropout_rate
        self.layernorm_eps = layernorm_eps
        
        # Calculate the number of patches and hidden sizes
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection_dim = kwargs.get("projection_dim", None)


class SiglipPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding module for the SigLIP vision encoder.
    
    This module:
    1. Divides the input image into patches (non-overlapping)
    2. Flattens each patch into a vector
    3. Projects each vector to the transformer's hidden dimension
    
    Example for a 224x224 image with 16x16 patches:
    - Number of patches: (224/16)² = 14² = 196
    - Each patch contains: 16²*3 = 768 values (for RGB image)
    - Output shape: (batch_size, 196, hidden_size)
    """
    def __init__(self, config):
        """
        Initialize the patch embedding layer.
        
        Args:
            config: SigLIP vision configuration
        """
        super().__init__()
        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size
        
        # Calculate number of patches (height × width)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Convolutional layer that:
        # - Takes patches of image_size with kernel_size=patch_size and stride=patch_size 
        # - Outputs hidden_size channels
        # - Effectively maps each patch to a vector of size hidden_size
        self.projection = nn.Conv2d(
            num_channels, hidden_size, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Convert images to patch embeddings.
        
        Args:
            pixel_values: Input images of shape (batch_size, channels, height, width)
            
        Returns:
            Patch embeddings of shape (batch_size, num_patches, hidden_size)
        """
        batch_size, num_channels, height, width = pixel_values.shape
        
        # Check if image dimensions are compatible with patch size
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model "
                f"configuration ({self.image_size}*{self.image_size})"
            )
        
        # Project patches using convolution
        # Shape: (batch_size, hidden_size, grid_h, grid_w)
        x = self.projection(pixel_values)
        
        # Rearrange to sequence of patches:
        # (batch_size, hidden_size, grid_h, grid_w) -> (batch_size, hidden_size, num_patches)
        # -> (batch_size, num_patches, hidden_size)
        patch_embeddings = x.flatten(2).transpose(1, 2)
        
        return patch_embeddings


class SiglipVisionEmbeddings(nn.Module):
    """
    Full embeddings for the SigLIP vision encoder.
    
    This combines:
    1. Patch embeddings (from dividing the image into patches)
    2. Class token embedding [CLS] (for image representation)
    3. Position embeddings (to encode patch positions)
    
    The output is a sequence of embedded patch+position tokens + [CLS] token
    that can be processed by the transformer encoder.
    """
    def __init__(self, config):
        """
        Initialize the vision embeddings module.
        
        Args:
            config: SigLIP vision configuration
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        
        # Patch embeddings (convert image patches to hidden dimension)
        self.patch_embeddings = SiglipPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        
        # Special learnable [CLS] token added to beginning of sequence
        # This token will aggregate information from all patches
        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        
        # Learnable position embeddings for each patch position + [CLS] token
        # Total positions: num_patches + 1 (for [CLS])
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.hidden_size)
        )
        
        # Learnable position embeddings for 2D positions of patches
        # Separate embeddings for each spatial dimension (height, width)
        self.position_ids = torch.arange(self.num_patches).expand((1, -1))
        
        # Pre-computed sin/cos position encodings
        self.register_buffer("position_ids_x", torch.arange(0, config.image_size // config.patch_size), persistent=False)
        self.register_buffer("position_ids_y", torch.arange(0, config.image_size // config.patch_size), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Process images into embedded sequences with positional information.
        
        Args:
            pixel_values: Input images of shape (batch_size, channels, height, width)
            
        Returns:
            Embedded sequence of shape (batch_size, num_patches+1, hidden_size)
        """
        batch_size = pixel_values.shape[0]
        
        # Convert image to patch embeddings
        # Shape: (batch_size, num_patches, hidden_size)
        patch_embeddings = self.patch_embeddings(pixel_values)
        
        # Expand class token for batch size and concatenate to beginning of sequence
        # Shape: (batch_size, 1, hidden_size) -> (batch_size, 1+num_patches, hidden_size)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeddings], dim=1)
        
        # Add position embeddings to patch+class embeddings
        # Shape: (batch_size, 1+num_patches, hidden_size)
        embeddings = embeddings + self.position_embedding
        
        return embeddings


class SiglipAttention(nn.Module):
    """
    Multi-head self-attention module for the SigLIP vision encoder.
    
    This implements standard transformer self-attention with:
    1. Linear projections for query, key, value
    2. Scaled dot-product attention
    3. Softmax normalization
    4. Optional attention dropout
    5. Output projection
    
    It computes: Attention(Q, K, V) = softmax(QK^T/√d_k)V
    """
    def __init__(self, config):
        """
        Initialize the attention module.
        
        Args:
            config: SigLIP vision configuration
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        
        # Check if hidden size is divisible by number of attention heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_attention_heads {self.num_attention_heads}"
            )
            
        # Calculate dimensions
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor for dot product
        
        # Linear projections for Q, K, V
        # All projections combined in one layer for efficiency
        self.qkv = nn.Linear(self.hidden_size, self.hidden_size * 3)
        
        # Output projection
        self.projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Attention dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply self-attention to input hidden states.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of:
            - Output tensor after attention
            - Attention weights (optional)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Combined projection for Q, K, V
        # Shape: (batch_size, seq_len, 3*hidden_size)
        qkv = self.qkv(hidden_states)
        
        # Reshape and split into Q, K, V
        # Shape: (batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_attention_heads, self.head_dim)
        
        # Permute to get separate Q, K, V tensors
        # Shape: 3 × (batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # Calculate scaled dot-product attention
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get attention weights
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention weights to values
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        context_layer = torch.matmul(attention_probs, v)
        
        # Reshape back to original format
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, hidden_size)
        context_layer = context_layer.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        
        # Apply output projection
        output = self.projection(context_layer)
        
        # Return output and optionally attention weights
        return (output, attention_probs) if output_attentions else (output, None)


class SiglipMLP(nn.Module):
    """
    MLP (Feed-Forward Network) for SigLIP vision encoder.
    
    This implements a two-layer feed-forward network with GELU activation
    that is applied after the self-attention layer in each transformer block.
    
    Structure: Linear → GELU → Linear
    """
    def __init__(self, config):
        """
        Initialize the MLP.
        
        Args:
            config: SigLIP vision configuration
        """
        super().__init__()
        self.config = config
        
        # Two-layer MLP
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Approximate GELU activation function (Gaussian Error Linear Unit)
        self.gelu = nn.GELU()
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply MLP to input hidden states.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor after feed-forward network
        """
        # First linear projection
        hidden_states = self.fc1(hidden_states)
        
        # GELU activation
        hidden_states = self.gelu(hidden_states)
        
        # Second linear projection
        hidden_states = self.fc2(hidden_states)
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    """
    Transformer encoder layer for SigLIP vision encoder.
    
    Each encoder layer consists of:
    1. Layer normalization
    2. Multi-head self-attention
    3. Residual connection
    4. Layer normalization
    5. MLP (feed-forward network)
    6. Residual connection
    
    This follows the "Pre-LN" transformer architecture where
    layer normalization is applied before attention and MLP.
    """
    def __init__(self, config):
        """
        Initialize the encoder layer.
        
        Args:
            config: SigLIP vision configuration
        """
        super().__init__()
        self.config = config
        
        # Layer normalization before attention (Pre-LN architecture)
        self.attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
        
        # Multi-head self-attention
        self.attention = SiglipAttention(config)
        
        # Layer normalization before MLP
        self.mlp_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
        
        # MLP block
        self.mlp = SiglipMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process inputs through the encoder layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of:
            - Output tensor after layer processing
            - Attention weights (optional)
        """
        # Pre-normalization for attention (Pre-LN architecture)
        attention_layernorm_output = self.attention_layernorm(hidden_states)
        
        # Apply self-attention
        attention_output, attention_weights = self.attention(
            hidden_states=attention_layernorm_output,
            output_attentions=output_attentions,
        )
        
        # First residual connection
        hidden_states = attention_output + hidden_states
        
        # Pre-normalization for MLP
        mlp_layernorm_output = self.mlp_layernorm(hidden_states)
        
        # Apply MLP
        mlp_output = self.mlp(mlp_layernorm_output)
        
        # Second residual connection
        hidden_states = mlp_output + hidden_states
        
        # Return output and optionally attention weights
        return (hidden_states, attention_weights) if output_attentions else (hidden_states, None)


class SiglipEncoder(nn.Module):
    """
    Transformer encoder for the SigLIP vision model.
    
    This consists of a stack of identical encoder layers that
    process the embedded image patches sequentially.
    """
    def __init__(self, config):
        """
        Initialize the encoder with multiple layers.
        
        Args:
            config: SigLIP vision configuration
        """
        super().__init__()
        self.config = config
        
        # Create a stack of identical encoder layers
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        
        # Final layer normalization
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_eps)
        
        # Initialize weights
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Tuple:
        """
        Process inputs through the transformer encoder.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            output_attentions: Whether to return attention weights for each layer
            output_hidden_states: Whether to return hidden states from each layer
            return_dict: Whether to return as dictionary or tuple
            
        Returns:
            Tuple containing:
            - Final hidden states after all layers
            - Hidden states from each layer (optional)
            - Attention weights from each layer (optional)
        """
        # Initialize lists to store intermediate states
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Process through each transformer layer
        for idx, encoder_layer in enumerate(self.layers):
            # Save hidden states if requested
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Process through the current encoder layer
            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                output_attentions=output_attentions,
            )
            
            # Update hidden states for next layer
            hidden_states = layer_outputs[0]
            
            # Save attention weights if requested
            if output_attentions:
                attention_weights = layer_outputs[1]
                all_attentions = all_attentions + (attention_weights,)
        
        # Apply final layer normalization
        hidden_states = self.layernorm(hidden_states)
        
        # Save final hidden states if requested
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        # Return results in the requested format
        if not return_dict:
            return (hidden_states, all_hidden_states, all_attentions)
            
        return hidden_states


class SiglipPooler(nn.Module):
    """
    Pooler for the SigLIP vision encoder.
    
    This extracts the [CLS] token embedding which serves as a global
    representation of the entire image.
    """
    def __init__(self, config):
        """
        Initialize the pooler.
        
        Args:
            config: SigLIP vision configuration
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Pool the encoder sequence to get image representation.
        
        Args:
            hidden_states: Final encoder hidden states
                           of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Pooled representation of shape (batch_size, hidden_size)
        """
        # Extract [CLS] token embedding from position 0 of the sequence
        # Shape: (batch_size, hidden_size)
        cls_token = hidden_states[:, 0]
        
        # Project and apply activation
        pooled_output = self.dense(cls_token)
        pooled_output = self.activation(pooled_output)
        
        return pooled_output


class SiglipVisionModel(nn.Module):
    """
    Complete SigLIP vision encoder model.
    
    This implements the full vision transformer used to encode images:
    1. Patch and position embeddings
    2. Transformer encoder layers
    3. Optional pooling to extract image representation
    
    The output can be either the full sequence of token embeddings
    or just the pooled [CLS] token representation.
    """
    def __init__(self, config):
        """
        Initialize the vision model.
        
        Args:
            config: SigLIP vision configuration
        """
        super().__init__()
        self.config = config
        
        # Image embedding layer
        self.embeddings = SiglipVisionEmbeddings(config)
        
        # Transformer encoder
        self.encoder = SiglipEncoder(config)
        
        # Optional pooler for [CLS] token
        self.pooler = SiglipPooler(config)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        Process images through the vision model.
        
        Args:
            pixel_values: Input images of shape (batch_size, channels, height, width)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return as dictionary or tuple
            
        Returns:
            Full sequence of encoded representations, excluding the [CLS] token
        """
        # Set default output options
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        # Ensure pixel values are provided
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        # Convert images to embedded sequence with positions
        embedding_output = self.embeddings(pixel_values)
        
        # Process embedded sequence through transformer encoder
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Extract relevant outputs
        if isinstance(encoder_outputs, tuple):
            last_hidden_state = encoder_outputs[0]
        else:
            last_hidden_state = encoder_outputs
        
        # Exclude the [CLS] token and return only patch embeddings
        return last_hidden_state[:, 1:]


# Test code to verify the implementation
if __name__ == "__main__":
    # Define configuration for testing
    config = SiglipVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=224,
        patch_size=16,
    )
    
    # Create model
    model = SiglipVisionModel(config)
    
    # Create dummy input (batch_size=2, channels=3, height=224, width=224)
    pixel_values = torch.randn(2, 3, 224, 224)
    
    # Run model
    outputs = model(pixel_values)
    
    # Check output shape (batch_size, num_patches, hidden_size)
    # num_patches = (image_size/patch_size)^2 = (224/16)^2 = 196
    print("Output shape:", outputs.shape)  # Expected: (2, 196, 768)
