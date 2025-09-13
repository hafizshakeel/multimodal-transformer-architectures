import math

import torch
from torch import nn
from typing import Optional, Tuple, List
from siglip import SiglipVisionConfig, SiglipVisionModel

"""
PaliGemma: Core implementation of the PaliGemma multimodal model.

This module implements the complete PaliGemma architecture, which consists of:
1. A SigLIP vision encoder for processing images
2. A multimodal projector to align vision and language representations
3. A Gemma language model for text generation conditioned on images and text

PaliGemma combines the SigLIP vision encoder with Gemma language model to enable
image understanding and text generation for a variety of multimodal tasks.
"""

class KVCache():
    """
    Key-Value Cache for efficient autoregressive generation.
    
    This class stores the key and value tensors for each transformer layer
    to avoid recomputing them for tokens that have already been processed.
    This significantly speeds up the generation process.
    """
    def __init__(self) -> None:
        """Initialize empty key and value cache lists."""
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        """
        Return the number of items (sequence length) in the cache.
        
        Returns:
            Number of cached tokens, or 0 if cache is empty
        """
        if len(self.key_cache) == 0:
            return 0
        else:
            # The sequence length is stored in the -2 dimension
            # Shape: (B, num_heads_kv, seq_len, head_dim)
            return self.key_cache[0].shape[-2]

    def update(
            self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new key and value states for a specific layer.
        
        For the first time a layer is processed, this adds the key/value states.
        For subsequent calls, it concatenates the new states with the cached ones.
        
        Args:
            key_states: New key states to add to the cache
            value_states: New value states to add to the cache
            layer_idx: Index of the transformer layer
            
        Returns:
            Tuple of the complete (cached + new) key and value states
        """
        if len(self.key_cache) <= layer_idx:
            # If nothing is added to the KVCache yet for this layer, add it
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Otherwise concatenate the new keys/values with the existing ones
            # This extends the sequence length dimension (-2)
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # Return all the existing keys/values and the new ones
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GemmaConfig():
    """
    Configuration class for the Gemma language model.
    
    This stores all hyperparameters required to build the language model
    components of the PaliGemma architecture.
    """
    def __init__(
        self,
        vocab_size,                     # Size of the vocabulary
        hidden_size,                    # Embedding dimension of tokens
        intermediate_size,              # Feed-forward network's hidden size
        num_hidden_layers,              # Number of transformer layers
        num_attention_heads,            # Number of attention heads
        num_key_value_heads,            # Number of key/value heads (for GQA)
        head_dim=256,                   # Dimension of each attention head
        max_position_embeddings=8192,   # Maximum sequence length for RoPE
        rms_norm_eps=1e-6,              # Epsilon for RMS normalization
        rope_theta=10000.0,             # Base frequency for RoPE
        attention_bias=False,           # Whether to use bias in attention projections
        attention_dropout=0.0,          # Dropout probability for attention
        pad_token_id=None,              # ID of padding token
        **kwargs,
    ):
        """Initialize Gemma configuration with model hyperparameters."""
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig():
    """
    Configuration class for the complete PaliGemma model.
    
    This combines configurations for both the vision and language components
    and handles coordination between them.
    """
    def __init__(
        self,
        vision_config=None,         # Configuration for the vision model
        text_config=None,           # Configuration for the language model
        ignore_index=-100,          # Index to ignore in loss computation
        image_token_index=256000,   # Token ID for <image> placeholder tokens
        vocab_size=257152,          # Size of the combined vocabulary
        projection_dim=2048,        # Output size of the vision-language projection
        hidden_size=2048,           # Embedding size of language model
        pad_token_id=None,          # ID of padding token
        **kwargs,
    ):
        """Initialize PaliGemma configuration combining vision and text models."""
        super().__init__()
        self.vision_config = vision_config
        self.text_config = text_config
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        # Create proper config objects from dictionaries
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        # Configure vision-language coordination parameters
        # Calculate number of image tokens based on patch size
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim 


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key and value states for Grouped Query Attention (GQA).
    
    In GQA, we have fewer key/value heads than query heads. This function
    repeats the key/value heads to match the number of query heads.
    
    Args:
        hidden_states: Key or value states with shape (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each head
        
    Returns:
        Repeated states with shape (batch, num_kv_heads * n_rep, seq_len, head_dim)
    """
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    
    # If no repetition needed, return as is
    if n_rep == 1:
        return hidden_states
        
    # Add a new dimension and expand along it
    # (B, num_kv_heads, 1, seq_len, head_dim) -> (B, num_kv_heads, n_rep, seq_len, head_dim)
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    
    # Reshape to merge the kv_heads and n_rep dimensions
    # (B, num_kv_heads, n_rep, seq_len, head_dim) -> (B, num_kv_heads * n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class GemmaRMSNorm(nn.Module):
    """
    Root Mean Square (RMS) Normalization layer used in Gemma.
    
    This is a variant of Layer Normalization that:
    1. Uses RMS (root mean square) instead of mean and variance
    2. Applies a learnable scale parameter but no bias
    
    RMS Norm tends to be more stable and computationally efficient.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMS normalization layer.
        
        Args:
            dim: Hidden dimension to normalize
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        # Initialize scale parameter to zero (will learn during training)
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """
        Apply RMS normalization without scaling.
        
        Formula: x / sqrt(mean(x²) + eps)
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Calculate root mean square along last dimension
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Apply RMS normalization with learned scaling.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized and scaled tensor
        """
        # Normalize in float32 for better numerical stability
        output = self._norm(x.float())
        
        # Apply learned per-dimension scaling (1 + weight)
        # The +1 ensures the layer can represent identity
        output = output * (1.0 + self.weight.float())
        
        # Convert back to input dtype
        return output.type_as(x)


class GemmaRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation for Gemma.
    
    RoPE encodes absolute positional information with a rotation matrix
    that naturally incorporates relative position information.
    
    Paper: https://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """
        Initialize rotary embeddings.
        
        Args:
            dim: Dimension of each attention head
            max_position_embeddings: Maximum sequence length
            base: Base value for frequency computation
            device: Computation device
        """
        super().__init__()

        self.dim = dim  # Set to the head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate the frequencies according to the RoPE formula: 
        # θᵢ = 10000^(-2i/d) where i = 0, 1, 2, ..., dim//2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        
        # Register as a buffer (not a parameter)
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        """
        Calculate cosine and sine embeddings for rotary position encoding.
        
        Args:
            x: Input tensor of shape [batch, num_heads, seq_len, head_dim]
            position_ids: Position indices
            seq_len: Optional sequence length
            
        Returns:
            Tuple of (cos, sin) tensors for rotary embeddings
        """
        # Move frequencies to the same device as input
        self.inv_freq.to(x.device)
        
        # Expand inverse frequencies for batch dimension
        # [dim/2] -> [batch, dim/2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        
        # Expand position IDs
        # [batch, seq_len] -> [batch, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Handle different device types (especially for MPS)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        
        # Disable autocast to ensure full precision for embeddings
        with torch.autocast(device_type=device_type, enabled=False):
            # Calculate frequencies for each position
            # [batch, dim/2, 1] @ [batch, 1, seq_len] -> [batch, dim/2, seq_len]
            # Then transpose to [batch, seq_len, dim/2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            
            # Duplicate frequencies to match head dimension
            # [batch, seq_len, dim/2] -> [batch, seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Calculate cos and sin for the rotation
            cos = emb.cos()
            sin = emb.sin()
            
        # Return cos and sin tensors with same dtype as input
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """
    Helper function for rotary embeddings that rotates half the dimensions.
    
    For a tensor [x₁, x₂, x₃, x₄, ...] this returns [-x₂, x₁, -x₄, x₃, ...]
    This is used in the RoPE calculation for the sine component.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with half the dimensions rotated
    """
    # Split the last dimension in half
    x1 = x[..., : x.shape[-1] // 2]  # First half
    x2 = x[..., x.shape[-1] // 2 :]  # Second half
    
    # Concatenate in the pattern [-x₂, x₁]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Apply rotary positional embeddings to query and key tensors.
    
    This implements the rotation matrix multiplication from the RoPE paper:
    [cos(mθ), -sin(mθ)]
    [sin(mθ),  cos(mθ)]
    
    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine part of rotary embedding
        sin: Sine part of rotary embedding
        unsqueeze_dim: Dimension to unsqueeze for broadcasting
        
    Returns:
        Tuple of rotated (q, k) tensors
    """
    # Add the head dimension for broadcasting
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Apply the rotation formula from the RoPE paper:
    # q_embed = q*cos + rotate_half(q)*sin
    # k_embed = k*cos + rotate_half(k)*sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class GemmaAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Embeddings for Gemma.
    
    This implements a variant of the standard Transformer attention with:
    1. Rotary Position Embeddings (RoPE) for handling positions
    2. Grouped Query Attention (GQA) for computational efficiency
    3. Support for key-value caching for efficient autoregressive generation
    """
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        """
        Initialize the attention module.
        
        Args:
            config: Model configuration
            layer_idx: Index of this layer in the transformer stack
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        
        # For GQA, we compute fewer key-value pairs than queries
        # This factor represents how many query heads share the same key-value head
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True  # Always use causal attention

        # Ensure hidden size is divisible by number of heads
        assert self.hidden_size % self.num_heads == 0

        # Projection matrices for queries, keys, values
        # For Grouped Query Attention (GQA), we use fewer key-value heads
        # See: https://arxiv.org/pdf/2305.13245
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        
        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # Rotary position embeddings
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> [torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Process inputs through multi-head attention with rotary embeddings.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Mask tensor to prevent attention to certain positions
            position_ids: Position indices for rotary embeddings
            kv_cache: Optional cache for key and value states
            
        Returns:
            Tuple containing:
            - Output tensor after attention
            - Attention weights (optional)
        """
        batch_size, q_len, _ = hidden_states.size()         # (B, seq_len, hidden_size)
        
        # Project inputs to queries, keys, and values
        query_states = self.q_proj(hidden_states)           # (B, seq_len, num_heads_q * head_dim)
        key_states = self.k_proj(hidden_states)             # (B, seq_len, num_heads_kv * head_dim)
        value_states = self.v_proj(hidden_states)           # (B, seq_len, num_heads_kv * head_dim)
        
        # Reshape for multi-head attention:
        # (B, seq_len, num_heads * head_dim) -> (B, num_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings for position encoding
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update KV cache for efficient autoregressive generation
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # For Grouped Query Attention, repeat key and value states to match number of query heads
        # This is a workaround for not having a custom CUDA kernel for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Calculate attention scores: Q * K^T / sqrt(d_k)
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3))) / math.sqrt(self.head_dim)
        # Shape: (B, num_heads_q, seq_len_q, seq_len_kv)

        # Ensure attention weights exist
        assert attn_weights is not None
        
        # Apply attention mask if provided
        attn_weights = attn_weights + attention_mask
        
        # Apply softmax to get normalized attention weights
        # Use float32 for numerical stability then cast back
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply dropout during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, value_states)  # (B, num_heads_q, seq_len_q, head_dim)

        # Verify output dimensions
        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)},"
                f" but it is of size {attn_output.size()}"
            )
            
        # Reshape back to original format:
        # (B, num_heads, seq_len, head_dim) -> (B, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Combine all heads:
        # (B, seq_len, num_heads, head_dim) -> (B, seq_len, hidden_size)
        attn_output = attn_output.view(batch_size, q_len, -1)
        
        # Final projection to output dimension
        attn_output = self.o_proj(attn_output)  # (B, seq_len, hidden_size)

        return attn_output, attn_weights


class GemmaMLP(nn.Module):
    """
    MLP (Feed-Forward Network) for Gemma model using SwiGLU activation.
    
    This implements the Gemma variant of the feed-forward network with:
    - Two parallel up-projections (gate and up)
    - GELU activation on the gate path
    - Element-wise multiplication of gate and up paths
    - Final down-projection
    
    This is a variant of SwiGLU: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, config: GemmaConfig):
        """
        Initialize the MLP.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Two parallel projections from hidden_size to intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        
        # Projection back to hidden_size
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        """
        Apply feed-forward network to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # SwiGLU-like activation:
        # down(GELU(gate(x)) * up(x))
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaDecoderLayer(nn.Module):
    """
    A single decoder layer (block) in the Gemma model.
    
    Each decoder layer consists of:
    1. Layer normalization
    2. Multi-head self-attention
    3. Residual connection
    4. Layer normalization
    5. MLP (feed-forward network)
    6. Residual connection
    
    This follows the standard Transformer decoder architecture with pre-norm design.
    """
    def __init__(self, config: GemmaConfig, layer_idx: int):
        """
        Initialize the decoder layer.
        
        Args:
            config: Model configuration
            layer_idx: Index of this layer in the transformer stack
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Pre-attention normalization
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Self-attention block
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        
        # Pre-MLP normalization
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # MLP block
        self.mlp = GemmaMLP(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> [torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Process inputs through the decoder layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Mask tensor to prevent attention to certain positions
            position_ids: Position indices for rotary embeddings
            kv_cache: Optional cache for key and value states
            
        Returns:
            Output tensor after passing through the decoder layer
        """
        # Store original input for first residual connection
        residual = hidden_states                                    # (B, seq_len, hidden_size)
        
        # Pre-normalization before attention (pre-norm architecture)
        hidden_states = self.input_layernorm(hidden_states)         # (B, seq_len, hidden_size)
        
        # Self-attention block
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            kv_cache=kv_cache
        )                                                           # (B, seq_len, hidden_size)
        
        # First residual connection
        hidden_states = residual + hidden_states                    # (B, seq_len, hidden_size)
        
        # Store for second residual connection
        residual = hidden_states                                    # (B, seq_len, hidden_size)
        
        # Pre-normalization before MLP
        hidden_states = self.post_attention_layernorm(hidden_states)    # (B, seq_len, hidden_size)
        
        # MLP (feed-forward) block
        hidden_states = self.mlp(hidden_states)                     # (B, seq_len, hidden_size)
        
        # Second residual connection
        hidden_states = residual + hidden_states                    # (B, seq_len, hidden_size)
        
        return hidden_states


class GemmaModel(nn.Module):
    """
    Complete Gemma language model architecture.
    
    This implements the core transformer model without the language modeling head.
    It consists of:
    1. Token embeddings
    2. A stack of decoder layers
    3. Final layer normalization
    """
    def __init__(self, config: GemmaConfig):
        """
        Initialize the Gemma model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embedding table
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Final layer normalization
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        """Return the token embedding module."""
        return self.embed_tokens

    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        """
        Process inputs through the Gemma model.
        
        Args:
            attention_mask: Mask tensor to prevent attention to certain positions
            position_ids: Position indices for rotary embeddings
            inputs_embeds: Pre-computed input embeddings
            kv_cache: Optional cache for key and value states
            
        Returns:
            Final hidden states after passing through the model
        """
        hidden_states = inputs_embeds  # (B, seq_len, hidden_size)
        
        # Scale embeddings by sqrt(hidden_size) for better training dynamics
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # Process through each decoder layer sequentially
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                kv_cache=kv_cache
            )

        # Apply final layer normalization
        hidden_states = self.norm(hidden_states)  # (B, seq_len, hidden_size)
        
        return hidden_states


class GemmaForCausalLM(nn.Module):
    """
    Gemma model with language modeling head for text generation.
    
    This wraps the GemmaModel and adds a linear layer to predict the next token.
    """
    def __init__(self, config):
        """
        Initialize the causal language model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Base transformer model
        self.model = GemmaModel(config)
        
        # Language modeling head (projects to vocabulary)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        """Return the token embedding module."""
        return self.model.embed_tokens

    def tie_weights(self):
        """Tie the weights between the input embeddings and LM head."""
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """
        Process inputs through the Gemma model and predict next tokens.
        
        Args:
            attention_mask: Mask tensor to prevent attention to certain positions
            position_ids: Position indices for rotary embeddings
            inputs_embeds: Pre-computed input embeddings
            kv_cache: Optional cache for key and value states
            
        Returns:
            Dictionary containing logits and other outputs
        """
        # Process through base model
        # inputs_embeds: (B, seq_len, hidden_size)
        # outputs: (B, seq_len, hidden_size)
        outputs = self.model(
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            inputs_embeds=inputs_embeds, 
            kv_cache=kv_cache
        )
        
        # Use final hidden states
        hidden_states = outputs
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        # Convert to float32 for better numerical stability in softmax
        logits = logits.float()

        # Prepare return data
        return_data = {
            "logits": logits,
        }

        # Include KV cache if provided
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data


class PaliGemmaMultiModalProjector(nn.Module):
    """
    Projection layer to align vision and language representations.
    
    This linear projection maps the vision encoder's output space
    to the language model's input space.
    """
    def __init__(self, config: PaliGemmaConfig):
        """
        Initialize the multimodal projector.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        # Linear projection from vision embedding space to language embedding space
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        """
        Project image features to language model embedding space.
        
        Args:
            image_features: Vision encoder output of shape (batch_size, num_patches, vision_dim)
            
        Returns:
            Projected features of shape (batch_size, num_patches, projection_dim)
        """
        # Project from vision hidden dimension to language model dimension
        # (B, num_patches, embed_dim) -> (B, num_patches, projection_dim)
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):
    """
    Complete PaliGemma model for multimodal conditional generation.
    
    This is the main class that combines:
    1. Vision encoder (SigLIP)
    2. Vision-language projector
    3. Language model (Gemma)
    
    It handles the processing of both image and text inputs and
    the fusion of these modalities for tasks like image captioning.
    """
    def __init__(self, config: PaliGemmaConfig):
        """
        Initialize the PaliGemma model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Three main components:
        self.vision_tower = SiglipVisionModel(config.vision_config)         # Vision Encoder
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)   # Linear Projection
        language_model = GemmaForCausalLM(config.text_config)               # Transformer Decoder
        self.language_model = language_model
        
        # Get pad token ID for masking
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        """Tie the weights between the input embeddings and LM head."""
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
            self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, 
            attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None
    ):
        """
        Merge image features and text embeddings into a unified representation.
        
        This function:
        1. Identifies positions of text tokens, image tokens, and padding
        2. Replaces image token placeholders with actual image features
        3. Creates appropriate attention masks and position IDs
        
        Args:
            image_features: Projected image features
            inputs_embeds: Text token embeddings
            input_ids: Input token IDs
            attention_mask: Attention mask for input
            kv_cache: Optional key-value cache
            
        Returns:
            Tuple of (merged_embeddings, attention_mask, position_ids)
        """
        # Extract dimensions
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        
        # Scale image features similar to how embeddings are scaled in transformer
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)
        
        # Initialize final embedding tensor
        final_embedding = torch.zeros(
            batch_size, sequence_length, embed_dim, 
            dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

        # Create masks to identify different token types:
        # - Text tokens: Regular text tokens (not image tokens or padding)
        # - Image tokens: Placeholders where image features should go
        # - Pad tokens: Padding tokens to ignore
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # Expand masks to match embedding dimension for element-wise operations
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # 1. Add text embeddings where text mask is True
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        
        # 2. Add image embeddings where image mask is True
        # (We use masked_scatter because image_features might have different sequence length)
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        
        # 3. Zero out padding token positions
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        # Create appropriate attention mask based on generation state
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Initial (prefill) phase: no masking needed
            # This assumes no padding in the input
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Generation phase: we're adding one token at a time
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Each query attends to all previous tokens
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add head dimension for attention
        # (B, q_len, kv_len) -> (B, 1, q_len, kv_len)
        causal_mask = causal_mask.unsqueeze(1)

        # Create position IDs based on attention mask and generation state
        if kv_cache is not None and kv_cache.num_items() > 0:
            # For generation, position of new token is the sequence length so far
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # For prefill, positions are sequential based on attention mask
            # Masked tokens get position 1 (a safe position)
            position_ids = attention_mask.cumsum(-1).masked_fill((attention_mask == 0), 1)
            
        return final_embedding, causal_mask, position_ids


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """
        Forward pass through the complete PaliGemma model.
        
        This processes both image and text inputs to generate text outputs.
        
        Args:
            input_ids: Token IDs for text input
            pixel_values: Pixel values from input image
            attention_mask: Attention mask for input
            kv_cache: Optional key-value cache
            
        Returns:
            Model outputs including logits
        """
        # Currently PaliGemma doesn't support padding in the input
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Get text token embeddings from embedding layer
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)  # (B, seq_len, hidden_size)

        # 2. Process image and merge with text
        # 2a. Encode image with vision tower
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))  # (B, num_patches, embed_dim)
        
        # 2b. Project vision features to language embedding space
        image_features = self.multi_modal_projector(selected_image_feature)  # (B, num_patches, projection_dim)
        
        # 2c. Merge text embeddings and image features
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, kv_cache
        )

        # 3. Pass merged embeddings through language model
        outputs = self.language_model(
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            inputs_embeds=inputs_embeds, 
            kv_cache=kv_cache
        )
        
        return outputs


# Test code to verify the implementation
if __name__ == "__main__":
    # Define configuration for vision model
    vision_config = {
        "image_size": 224,           # Input image resolution
        "patch_size": 16,            # Size of image patches
        "num_channels": 3,           # RGB images
        "hidden_size": 1024,         # Vision encoder hidden dimension
        "num_hidden_layers": 24,     # Number of transformer layers
        "num_attention_heads": 16,   # Number of attention heads
        "intermediate_size": 4096,   # Size of MLP layers
        "layernorm_eps": 1e-6,       # Layer norm epsilon
        "dropout_rate": 0.0,         # Dropout rate
        "attention_dropout_rate": 0.0,  # Attention dropout rate
    }

    # Define configuration for language model
    text_config = {
        "vocab_size": 257152,           # Size of token vocabulary
        "hidden_size": 2048,            # Hidden dimension
        "intermediate_size": 8192,      # Size of MLP layers
        "num_hidden_layers": 24,        # Number of transformer layers
        "num_attention_heads": 16,      # Number of attention heads
        "num_key_value_heads": 8,       # Number of key/value heads (GQA)
        "head_dim": 128,                # Dimension of each head
        "max_position_embeddings": 8192,  # Maximum sequence length
        "rms_norm_eps": 1e-6,           # RMS norm epsilon
        "rope_theta": 100000.0,         # RoPE base
        "attention_bias": False,        # No bias in attention
        "attention_dropout": 0.0,       # No dropout in attention
    }

    # Create full PaliGemma configuration
    config = PaliGemmaConfig(
        vision_config=vision_config,
        text_config=text_config,
        ignore_index=-100,
        image_token_index=256000,        # ID for <image> token
        vocab_size=257152,
        projection_dim=2048,             # Vision-to-language projection dimension
        hidden_size=2048,
        pad_token_id=1
    )

    # Create model
    model = PaliGemmaForConditionalGeneration(config)

    # Create dummy inputs
    batch_size = 2
    num_channels = 3
    image_size = 224
    seq_length = 10

    pixel_values = torch.randn(batch_size, num_channels, image_size, image_size)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    # Run model forward pass
    outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
    print("Logits shape:", outputs["logits"].shape)  # Expected: (batch_size, seq_length, vocab_size)

