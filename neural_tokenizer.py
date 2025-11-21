#!/usr/bin/env python3
"""
Neural Signal Tokenizer for Ghost BCI

Converts continuous BCI signals into discrete tokens for efficient processing:
    - VQ-VAE based tokenization (like audio codecs)
    - Multi-scale temporal encoding
    - Spatial (channel) tokenization
    - Frequency band tokenization
    - Hierarchical codebooks
    - Streaming tokenization

This is analogous to text tokenization for LLMs, but for neural signals.

Author: Claude (Anthropic) in collaboration with John Sayers
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class TokenizerConfig:
    """Configuration for neural tokenizer."""

    # Input
    num_channels: int = 64
    sample_rate: int = 250
    window_size: int = 250  # 1 second

    # Tokenization
    vocab_size: int = 8192
    num_codebooks: int = 4  # Hierarchical RVQ
    codebook_dim: int = 256

    # Encoder
    encoder_dim: int = 512
    encoder_layers: int = 4
    encoder_heads: int = 8

    # Temporal
    temporal_downsample: int = 4  # Downsample factor
    tokens_per_second: int = 50

    # Frequency bands
    use_frequency_bands: bool = True
    frequency_bands: List[Tuple[float, float]] = None

    # Training
    commitment_weight: float = 0.25
    ema_decay: float = 0.99

    def __post_init__(self):
        if self.frequency_bands is None:
            self.frequency_bands = [
                (0.5, 4),    # Delta
                (4, 8),      # Theta
                (8, 13),     # Alpha
                (13, 30),    # Beta
                (30, 100),   # Gamma
            ]


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with EMA updates.

    Converts continuous vectors to discrete codebook indices.
    """

    def __init__(
        self,
        vocab_size: int,
        codebook_dim: int,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay

        # Codebook embeddings
        self.embedding = nn.Embedding(vocab_size, codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

        # EMA cluster tracking
        self.register_buffer('ema_cluster_size', torch.zeros(vocab_size))
        self.register_buffer('ema_w', torch.zeros(vocab_size, codebook_dim))

        # Initialize EMA weights
        self.ema_w.data.copy_(self.embedding.weight.data)

    def forward(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input vectors.

        Args:
            z: Input tensor [batch, seq, dim]

        Returns:
            Tuple of (quantized, indices, loss)
        """
        # Flatten for distance computation
        flat_z = z.reshape(-1, self.codebook_dim)

        # Compute distances
        distances = (
            (flat_z ** 2).sum(dim=1, keepdim=True)
            + (self.embedding.weight ** 2).sum(dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
        )

        # Get nearest codebook entries
        encoding_indices = distances.argmin(dim=1)
        encodings = F.one_hot(encoding_indices, self.vocab_size).float()

        # Quantize
        quantized = self.embedding(encoding_indices)
        quantized = quantized.view_as(z)

        # EMA update (only in training)
        if self.training:
            self._ema_update(flat_z, encodings)

        # Compute loss
        commitment_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Reshape indices
        indices = encoding_indices.view(z.shape[:-1])

        return quantized, indices, loss

    def _ema_update(self, flat_z: torch.Tensor, encodings: torch.Tensor):
        """Update codebook with EMA."""
        # Cluster size
        n = encodings.sum(dim=0)
        self.ema_cluster_size.data.mul_(self.ema_decay).add_(
            n, alpha=1 - self.ema_decay
        )

        # Sum of vectors in each cluster
        dw = torch.matmul(encodings.t(), flat_z)
        self.ema_w.data.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

        # Normalize
        n = self.ema_cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(self.ema_w / (n + 1e-5))

    def get_codebook_usage(self) -> float:
        """Get fraction of codebook being used."""
        used = (self.ema_cluster_size > 0).sum().item()
        return used / self.vocab_size


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ).

    Uses multiple codebooks in sequence, each encoding the residual
    from the previous. Gives better reconstruction than single VQ.
    """

    def __init__(
        self,
        num_codebooks: int,
        vocab_size: int,
        codebook_dim: int,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks

        self.quantizers = nn.ModuleList([
            VectorQuantizer(vocab_size, codebook_dim, commitment_weight, ema_decay)
            for _ in range(num_codebooks)
        ])

    def forward(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize with multiple residual codebooks.

        Args:
            z: Input tensor [batch, seq, dim]

        Returns:
            Tuple of (quantized, indices, loss)
        """
        quantized = torch.zeros_like(z)
        residual = z
        all_indices = []
        total_loss = 0

        for quantizer in self.quantizers:
            q, indices, loss = quantizer(residual)
            quantized = quantized + q
            residual = residual - q
            all_indices.append(indices)
            total_loss = total_loss + loss

        # Stack indices: [num_codebooks, batch, seq]
        indices = torch.stack(all_indices, dim=0)

        return quantized, indices, total_loss / self.num_codebooks

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from indices.

        Args:
            indices: Token indices [num_codebooks, batch, seq]

        Returns:
            Reconstructed vectors [batch, seq, dim]
        """
        quantized = None

        for i, quantizer in enumerate(self.quantizers):
            q = quantizer.embedding(indices[i])
            if quantized is None:
                quantized = q
            else:
                quantized = quantized + q

        return quantized


class FrequencyBandEncoder(nn.Module):
    """
    Encode different frequency bands separately.

    Applies bandpass filtering and encodes each band,
    then combines for richer representation.
    """

    def __init__(
        self,
        num_channels: int,
        sample_rate: int,
        bands: List[Tuple[float, float]],
        output_dim: int,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.bands = bands
        self.num_bands = len(bands)

        # Per-band encoders
        self.band_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_channels, output_dim // self.num_bands, 3, padding=1),
                nn.GELU(),
                nn.Conv1d(output_dim // self.num_bands, output_dim // self.num_bands, 3, padding=1),
            )
            for _ in bands
        ])

        # Combine bands
        self.combine = nn.Linear(output_dim, output_dim)

    def bandpass_filter(
        self,
        x: torch.Tensor,
        low: float,
        high: float
    ) -> torch.Tensor:
        """Apply bandpass filter using FFT."""
        # FFT
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0/self.sample_rate)
        freqs = freqs.to(x.device)

        # Create bandpass mask
        mask = ((freqs >= low) & (freqs <= high)).float()

        # Apply mask
        X = X * mask.unsqueeze(0).unsqueeze(0)

        # Inverse FFT
        return torch.fft.irfft(X, n=x.shape[-1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode frequency bands.

        Args:
            x: Input [batch, channels, samples]

        Returns:
            Encoded [batch, samples, output_dim]
        """
        band_features = []

        for i, (low, high) in enumerate(self.bands):
            # Filter
            filtered = self.bandpass_filter(x, low, high)
            # Encode
            encoded = self.band_encoders[i](filtered)
            band_features.append(encoded)

        # Concatenate bands
        combined = torch.cat(band_features, dim=1)  # [batch, output_dim, samples]
        combined = combined.transpose(1, 2)  # [batch, samples, output_dim]

        # Mix
        return self.combine(combined)


class SpatialEncoder(nn.Module):
    """
    Encode spatial (channel) patterns.

    Learns spatial relationships between EEG channels
    using attention over channel dimension.
    """

    def __init__(
        self,
        num_channels: int,
        embed_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_channels = num_channels

        # Channel embeddings (learnable positions)
        self.channel_embed = nn.Parameter(torch.randn(1, num_channels, embed_dim))

        # Attention over channels
        self.channel_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode spatial patterns.

        Args:
            x: Input [batch, channels, samples]

        Returns:
            Encoded [batch, samples, embed_dim]
        """
        batch_size, num_channels, num_samples = x.shape

        # Transpose to [batch, samples, channels]
        x = x.transpose(1, 2)

        # Add channel embeddings
        x = x.unsqueeze(2) * self.channel_embed.unsqueeze(1)
        x = x.view(batch_size * num_samples, num_channels, -1)

        # Attention over channels
        attn_out, _ = self.channel_attn(x, x, x)

        # Pool over channels
        x = attn_out.mean(dim=1)  # [batch*samples, embed_dim]
        x = x.view(batch_size, num_samples, -1)

        return self.proj(x)


class TemporalDownsampler(nn.Module):
    """
    Downsample temporal dimension while preserving information.

    Uses strided convolutions with learned downsampling.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        downsample_factor: int = 4,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        # Progressive downsampling
        layers = []
        current_dim = in_dim

        while downsample_factor > 1:
            stride = min(2, downsample_factor)
            downsample_factor = downsample_factor // stride

            layers.append(
                nn.Conv1d(current_dim, out_dim, kernel_size=3, stride=stride, padding=1)
            )
            layers.append(nn.GELU())
            current_dim = out_dim

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample temporal dimension.

        Args:
            x: Input [batch, seq, dim]

        Returns:
            Downsampled [batch, seq//factor, dim]
        """
        x = x.transpose(1, 2)  # [batch, dim, seq]
        x = self.layers(x)
        return x.transpose(1, 2)  # [batch, seq, dim]


class NeuralTokenizer(nn.Module):
    """
    Complete neural signal tokenizer.

    Converts continuous BCI signals into discrete tokens using:
    - Frequency band encoding
    - Spatial pattern encoding
    - Temporal downsampling
    - Residual vector quantization
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config

        # Frequency band encoder
        if config.use_frequency_bands:
            self.freq_encoder = FrequencyBandEncoder(
                config.num_channels,
                config.sample_rate,
                config.frequency_bands,
                config.encoder_dim,
            )
        else:
            self.freq_encoder = None
            # Simple projection instead
            self.input_proj = nn.Conv1d(
                config.num_channels, config.encoder_dim, 1
            )

        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(
            config.num_channels,
            config.encoder_dim,
            config.encoder_heads,
        )

        # Combine frequency and spatial
        self.combine = nn.Linear(config.encoder_dim * 2, config.encoder_dim)

        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_dim,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_dim * 4,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.encoder_layers
        )

        # Temporal downsampling
        self.downsampler = TemporalDownsampler(
            config.encoder_dim,
            config.codebook_dim,
            config.temporal_downsample,
        )

        # Vector quantization
        self.quantizer = ResidualVectorQuantizer(
            config.num_codebooks,
            config.vocab_size,
            config.codebook_dim,
            config.commitment_weight,
            config.ema_decay,
        )

        # Decoder (for reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(config.codebook_dim, config.encoder_dim),
            nn.GELU(),
            nn.Linear(config.encoder_dim, config.num_channels * config.temporal_downsample),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode BCI signal to tokens.

        Args:
            x: Input signal [batch, channels, samples]

        Returns:
            Tuple of (quantized, token_indices, vq_loss)
        """
        # Frequency encoding
        if self.freq_encoder is not None:
            freq_features = self.freq_encoder(x)
        else:
            freq_features = self.input_proj(x).transpose(1, 2)

        # Spatial encoding
        spatial_features = self.spatial_encoder(x)

        # Combine
        combined = torch.cat([freq_features, spatial_features], dim=-1)
        combined = self.combine(combined)

        # Temporal encoding
        encoded = self.temporal_encoder(combined)

        # Downsample
        downsampled = self.downsampler(encoded)

        # Quantize
        quantized, indices, vq_loss = self.quantizer(downsampled)

        return quantized, indices, vq_loss

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens back to signal.

        Args:
            indices: Token indices [num_codebooks, batch, seq]

        Returns:
            Reconstructed signal [batch, channels, samples]
        """
        # Get quantized vectors
        quantized = self.quantizer.decode(indices)

        # Decode
        decoded = self.decoder(quantized)

        # Reshape to channels and samples
        batch_size = decoded.shape[0]
        num_tokens = decoded.shape[1]

        # [batch, tokens, channels * downsample] -> [batch, channels, samples]
        decoded = decoded.view(
            batch_size, num_tokens,
            self.config.num_channels,
            self.config.temporal_downsample
        )
        decoded = decoded.permute(0, 2, 1, 3).contiguous()
        decoded = decoded.view(batch_size, self.config.num_channels, -1)

        return decoded

    def forward(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode and decode.

        Args:
            x: Input signal [batch, channels, samples]

        Returns:
            Dictionary with tokens, reconstruction, and losses
        """
        # Encode
        quantized, indices, vq_loss = self.encode(x)

        # Decode
        reconstructed = self.decode(indices)

        # Reconstruction loss (truncate to match)
        min_len = min(x.shape[-1], reconstructed.shape[-1])
        recon_loss = F.mse_loss(reconstructed[..., :min_len], x[..., :min_len])

        return {
            'tokens': indices,
            'quantized': quantized,
            'reconstructed': reconstructed,
            'vq_loss': vq_loss,
            'recon_loss': recon_loss,
            'loss': vq_loss + recon_loss,
        }

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize BCI signal (encode only).

        Args:
            x: Input signal [batch, channels, samples]

        Returns:
            Token indices [num_codebooks, batch, num_tokens]
        """
        _, indices, _ = self.encode(x)
        return indices

    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert tokens back to signal.

        Args:
            indices: Token indices [num_codebooks, batch, num_tokens]

        Returns:
            Reconstructed signal [batch, channels, samples]
        """
        return self.decode(indices)


class StreamingTokenizer:
    """
    Streaming tokenizer for real-time BCI processing.

    Maintains a buffer and tokenizes as data arrives.
    """

    def __init__(
        self,
        tokenizer: NeuralTokenizer,
        buffer_size: int = 1000,
        hop_size: int = 250,
        device: str = 'cuda',
    ):
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.hop_size = hop_size
        self.device = device

        # Initialize buffer
        self.buffer = None
        self.buffer_pos = 0

        # Token output
        self.token_buffer = []

    def reset(self, num_channels: int):
        """Reset the streaming state."""
        self.buffer = torch.zeros(
            1, num_channels, self.buffer_size,
            device=self.device
        )
        self.buffer_pos = 0
        self.token_buffer = []

    def add_samples(
        self,
        samples: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Add samples and return tokens if available.

        Args:
            samples: New samples [channels, num_samples] or [num_samples, channels]

        Returns:
            Token indices if a chunk was tokenized, else None
        """
        if self.buffer is None:
            self.reset(samples.shape[0] if samples.dim() == 2 else samples.shape[1])

        # Ensure correct shape
        if samples.dim() == 2:
            if samples.shape[1] == self.buffer.shape[1]:
                samples = samples.t()  # [num_samples, channels] -> [channels, samples]
        samples = samples.to(self.device)

        num_new = samples.shape[-1]

        # Add to buffer
        if self.buffer_pos + num_new <= self.buffer_size:
            self.buffer[0, :, self.buffer_pos:self.buffer_pos + num_new] = samples
            self.buffer_pos += num_new
        else:
            # Shift buffer
            shift = num_new
            self.buffer[0, :, :-shift] = self.buffer[0, :, shift:].clone()
            self.buffer[0, :, -shift:] = samples[:, -shift:]
            self.buffer_pos = self.buffer_size

        # Tokenize if we have enough
        if self.buffer_pos >= self.hop_size:
            tokens = self._tokenize_chunk()
            return tokens

        return None

    def _tokenize_chunk(self) -> torch.Tensor:
        """Tokenize the current buffer."""
        with torch.no_grad():
            tokens = self.tokenizer.tokenize(self.buffer)

        self.token_buffer.append(tokens)
        return tokens

    def get_all_tokens(self) -> torch.Tensor:
        """Get all tokens generated so far."""
        if not self.token_buffer:
            return None
        return torch.cat(self.token_buffer, dim=-1)


def create_neural_tokenizer(
    num_channels: int = 64,
    sample_rate: int = 250,
    vocab_size: int = 8192,
    num_codebooks: int = 4,
) -> NeuralTokenizer:
    """
    Create a neural tokenizer with default settings.

    Args:
        num_channels: Number of BCI channels
        sample_rate: Sampling rate in Hz
        vocab_size: Size of token vocabulary
        num_codebooks: Number of RVQ codebooks

    Returns:
        Configured neural tokenizer
    """
    config = TokenizerConfig(
        num_channels=num_channels,
        sample_rate=sample_rate,
        vocab_size=vocab_size,
        num_codebooks=num_codebooks,
    )

    return NeuralTokenizer(config)


# Example usage
if __name__ == "__main__":
    print("Ghost BCI Neural Tokenizer")
    print("=" * 50)

    # Create tokenizer
    config = TokenizerConfig(
        num_channels=64,
        sample_rate=250,
        vocab_size=4096,
        num_codebooks=4,
        encoder_dim=256,
        encoder_layers=2,
    )

    tokenizer = NeuralTokenizer(config)

    # Count parameters
    params = sum(p.numel() for p in tokenizer.parameters())
    print(f"Parameters: {params:,}")

    # Test tokenization
    print("\nTesting tokenization...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = tokenizer.to(device)

    # Simulate BCI data
    batch_size = 2
    num_samples = 250  # 1 second
    x = torch.randn(batch_size, config.num_channels, num_samples, device=device)

    # Forward pass
    output = tokenizer(x)

    print(f"Input shape: {x.shape}")
    print(f"Token shape: {output['tokens'].shape}")
    print(f"Reconstructed shape: {output['reconstructed'].shape}")
    print(f"VQ loss: {output['vq_loss'].item():.4f}")
    print(f"Recon loss: {output['recon_loss'].item():.4f}")

    # Test streaming
    print("\nTesting streaming tokenization...")
    streaming = StreamingTokenizer(tokenizer, buffer_size=250, hop_size=50, device=device)
    streaming.reset(config.num_channels)

    # Simulate streaming
    total_tokens = 0
    for i in range(10):
        chunk = torch.randn(config.num_channels, 50, device=device)
        tokens = streaming.add_samples(chunk)
        if tokens is not None:
            total_tokens += tokens.shape[-1]
            print(f"  Chunk {i+1}: Generated {tokens.shape[-1]} tokens")

    print(f"Total tokens generated: {total_tokens}")

    # Codebook usage
    usage = tokenizer.quantizer.quantizers[0].get_codebook_usage()
    print(f"\nCodebook usage: {usage*100:.1f}%")

    print("\nNeural tokenizer ready!")
