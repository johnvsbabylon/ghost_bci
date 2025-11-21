#!/usr/bin/env python3
"""
Optimized Inference Engine for Ghost BCI MoE

Production-grade inference optimizations for real-time BCI processing:
    - KV cache for efficient sequential processing
    - Dynamic batching for throughput
    - Quantization support (INT8/INT4)
    - Streaming inference for real-time applications
    - Memory-efficient attention
    - CUDA graph optimization
    - Speculative decoding for neural patterns

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
import time
from collections import deque
import threading
import queue

try:
    from ghost_bci_moe import GhostBCIMoE, GhostBCIMoEConfig
except ImportError:
    GhostBCIMoE = None
    GhostBCIMoEConfig = None


@dataclass
class InferenceConfig:
    """Configuration for optimized inference."""

    # Cache settings
    max_cache_length: int = 8192
    cache_dtype: torch.dtype = torch.float16

    # Batching
    max_batch_size: int = 32
    dynamic_batching: bool = True
    batch_timeout_ms: float = 10.0

    # Quantization
    quantization: Optional[str] = None  # None, 'int8', 'int4'

    # Memory optimization
    use_flash_attention: bool = True
    memory_efficient_attention: bool = True
    offload_to_cpu: bool = False

    # CUDA optimization
    use_cuda_graphs: bool = False
    cudagraph_warmup_steps: int = 3

    # Streaming
    stream_chunk_size: int = 64
    prefetch_chunks: int = 2

    # Speculative decoding
    use_speculative: bool = False
    draft_model_path: Optional[str] = None
    speculative_tokens: int = 4


class KVCache:
    """
    Key-Value cache for efficient sequential processing.

    Stores past key/value tensors to avoid recomputation during
    autoregressive generation or streaming inference.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_length: int,
        dtype: torch.dtype = torch.float16,
        device: str = 'cuda'
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_length = max_length
        self.dtype = dtype
        self.device = device

        # Pre-allocate cache tensors
        # Shape: [num_layers, 2, batch_size, num_heads, max_length, head_dim]
        # 2 is for key and value
        self.cache = None
        self.current_length = 0
        self.batch_size = 0

    def initialize(self, batch_size: int):
        """Initialize cache for a given batch size."""
        self.batch_size = batch_size
        self.current_length = 0

        # Allocate tensors
        self.cache = torch.zeros(
            self.num_layers, 2, batch_size, self.num_heads,
            self.max_length, self.head_dim,
            dtype=self.dtype, device=self.device
        )

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value and return full cached key/value.

        Args:
            layer_idx: Which layer's cache to update
            key: New key tensor [batch, heads, seq_len, head_dim]
            value: New value tensor [batch, heads, seq_len, head_dim]

        Returns:
            Tuple of (cached_key, cached_value) including new values
        """
        seq_len = key.shape[2]

        # Store in cache
        start_idx = self.current_length
        end_idx = start_idx + seq_len

        if end_idx > self.max_length:
            # Shift cache to make room (sliding window)
            shift = end_idx - self.max_length
            self.cache[:, :, :, :, :-shift, :] = self.cache[:, :, :, :, shift:, :].clone()
            start_idx -= shift
            end_idx = self.max_length
            self.current_length = self.max_length - seq_len

        self.cache[layer_idx, 0, :, :, start_idx:end_idx, :] = key
        self.cache[layer_idx, 1, :, :, start_idx:end_idx, :] = value

        # Return cached values up to current position
        new_length = self.current_length + seq_len
        cached_key = self.cache[layer_idx, 0, :, :, :new_length, :]
        cached_value = self.cache[layer_idx, 1, :, :, :new_length, :]

        # Update length (done after layer 0 only to avoid multiple updates)
        if layer_idx == 0:
            self.current_length = new_length

        return cached_key, cached_value

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached key/value for a layer."""
        return (
            self.cache[layer_idx, 0, :, :, :self.current_length, :],
            self.cache[layer_idx, 1, :, :, :self.current_length, :]
        )

    def clear(self):
        """Clear the cache."""
        self.current_length = 0
        if self.cache is not None:
            self.cache.zero_()


class QuantizedLinear(nn.Module):
    """INT8/INT4 quantized linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        group_size: int = 128
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        # Quantized weights
        if bits == 8:
            self.register_buffer('weight_quantized',
                torch.zeros(out_features, in_features, dtype=torch.int8))
        else:  # 4-bit packed into int8
            self.register_buffer('weight_quantized',
                torch.zeros(out_features, in_features // 2, dtype=torch.int8))

        # Scales per group
        num_groups = (in_features + group_size - 1) // group_size
        self.register_buffer('scales', torch.ones(out_features, num_groups))
        self.register_buffer('zeros', torch.zeros(out_features, num_groups))

    @staticmethod
    def from_float(linear: nn.Linear, bits: int = 8, group_size: int = 128):
        """Convert a float linear layer to quantized."""
        quant = QuantizedLinear(
            linear.in_features, linear.out_features, bits, group_size
        )

        weight = linear.weight.data

        # Quantize per group
        num_groups = (linear.in_features + group_size - 1) // group_size

        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, linear.in_features)
            group_weight = weight[:, start:end]

            # Compute scale and zero point
            w_min = group_weight.min(dim=1, keepdim=True)[0]
            w_max = group_weight.max(dim=1, keepdim=True)[0]

            if bits == 8:
                scale = (w_max - w_min) / 255
                zero = -w_min / scale
            else:  # 4-bit
                scale = (w_max - w_min) / 15
                zero = -w_min / scale

            quant.scales[:, g:g+1] = scale
            quant.zeros[:, g:g+1] = zero

            # Quantize
            q_weight = torch.round((group_weight - w_min) / scale).to(torch.int8)

            if bits == 8:
                quant.weight_quantized[:, start:end] = q_weight
            else:
                # Pack 4-bit values
                if (end - start) % 2 == 0:
                    packed_start = start // 2
                    packed_end = end // 2
                    low = q_weight[:, ::2]
                    high = q_weight[:, 1::2]
                    packed = (high << 4) | (low & 0x0F)
                    quant.weight_quantized[:, packed_start:packed_end] = packed

        return quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize and compute matmul."""
        # Dequantize weight
        if self.bits == 8:
            weight = self.weight_quantized.float()
        else:
            # Unpack 4-bit
            packed = self.weight_quantized
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            weight = torch.stack([low, high], dim=-1).reshape(
                self.out_features, self.in_features
            ).float()

        # Apply scales
        num_groups = self.scales.shape[1]
        group_size = (self.in_features + num_groups - 1) // num_groups

        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, self.in_features)
            weight[:, start:end] = (
                weight[:, start:end] - self.zeros[:, g:g+1]
            ) * self.scales[:, g:g+1]

        return F.linear(x, weight)


class StreamingInference:
    """
    Streaming inference for real-time BCI processing.

    Processes BCI data in chunks as it arrives, maintaining
    temporal context through the KV cache.
    """

    def __init__(
        self,
        model: nn.Module,
        config: InferenceConfig,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device

        # Get model config
        if hasattr(model, 'config'):
            model_config = model.config
            self.num_layers = model_config.num_layers
            self.num_heads = model_config.num_heads
            self.head_dim = model_config.embed_dim // model_config.num_heads
        else:
            # Defaults
            self.num_layers = 32
            self.num_heads = 32
            self.head_dim = 128

        # Initialize KV cache
        self.kv_cache = KVCache(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_length=config.max_cache_length,
            dtype=config.cache_dtype,
            device=device
        )

        # Chunk buffer
        self.chunk_buffer = deque(maxlen=config.prefetch_chunks)
        self.output_queue = queue.Queue()

        # Processing state
        self.is_streaming = False
        self.process_thread = None

    def start_stream(self, batch_size: int = 1):
        """Start streaming inference."""
        self.kv_cache.initialize(batch_size)
        self.is_streaming = True

        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()

    def stop_stream(self):
        """Stop streaming inference."""
        self.is_streaming = False
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
        self.kv_cache.clear()

    def add_chunk(self, chunk: torch.Tensor):
        """Add a chunk of BCI data to process."""
        self.chunk_buffer.append(chunk)

    def get_output(self, timeout: float = 0.1) -> Optional[Dict[str, torch.Tensor]]:
        """Get processed output if available."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _process_loop(self):
        """Background processing loop."""
        while self.is_streaming:
            if self.chunk_buffer:
                chunk = self.chunk_buffer.popleft()
                output = self._process_chunk(chunk)
                self.output_queue.put(output)
            else:
                time.sleep(0.001)

    def _process_chunk(self, chunk: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process a single chunk with KV cache."""
        with torch.no_grad():
            with autocast(dtype=self.config.cache_dtype):
                # Run through model with cache
                output = self.model(
                    chunk.to(self.device),
                    kv_cache=self.kv_cache,
                    use_cache=True
                )
        return output


class DynamicBatcher:
    """
    Dynamic batching for throughput optimization.

    Collects requests and batches them together for
    efficient GPU utilization.
    """

    def __init__(
        self,
        model: nn.Module,
        config: InferenceConfig,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device

        # Request queue
        self.request_queue = queue.Queue()
        self.result_futures: Dict[int, Any] = {}

        # Batching state
        self.is_running = False
        self.batch_thread = None
        self.request_id = 0

    def start(self):
        """Start the batcher."""
        self.is_running = True
        self.batch_thread = threading.Thread(target=self._batch_loop)
        self.batch_thread.daemon = True
        self.batch_thread.start()

    def stop(self):
        """Stop the batcher."""
        self.is_running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=1.0)

    def submit(self, input_tensor: torch.Tensor) -> int:
        """Submit a request and get a request ID."""
        request_id = self.request_id
        self.request_id += 1

        self.result_futures[request_id] = threading.Event()
        self.request_queue.put((request_id, input_tensor))

        return request_id

    def get_result(self, request_id: int, timeout: float = 10.0) -> Optional[Dict[str, torch.Tensor]]:
        """Get result for a request ID."""
        if request_id not in self.result_futures:
            return None

        event = self.result_futures[request_id]
        if event.wait(timeout):
            result = self.result_futures.pop(f"{request_id}_result", None)
            self.result_futures.pop(request_id, None)
            return result
        return None

    def _batch_loop(self):
        """Main batching loop."""
        while self.is_running:
            # Collect requests
            batch = []
            request_ids = []

            # Wait for at least one request
            try:
                req_id, tensor = self.request_queue.get(timeout=0.1)
                batch.append(tensor)
                request_ids.append(req_id)
            except queue.Empty:
                continue

            # Try to fill batch
            deadline = time.time() + self.config.batch_timeout_ms / 1000
            while len(batch) < self.config.max_batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break

                try:
                    req_id, tensor = self.request_queue.get(timeout=remaining)
                    batch.append(tensor)
                    request_ids.append(req_id)
                except queue.Empty:
                    break

            # Process batch
            if batch:
                self._process_batch(batch, request_ids)

    def _process_batch(self, batch: List[torch.Tensor], request_ids: List[int]):
        """Process a batch of requests."""
        # Pad and stack
        max_len = max(t.shape[-1] for t in batch)
        padded = []
        for t in batch:
            if t.shape[-1] < max_len:
                pad_size = max_len - t.shape[-1]
                t = F.pad(t, (0, pad_size))
            padded.append(t)

        batched = torch.stack(padded).to(self.device)

        # Run inference
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                outputs = self.model(batched)

        # Distribute results
        for i, req_id in enumerate(request_ids):
            result = {k: v[i:i+1] for k, v in outputs.items()}
            self.result_futures[f"{req_id}_result"] = result
            self.result_futures[req_id].set()


class OptimizedInferenceEngine:
    """
    Main inference engine with all optimizations.

    Features:
    - KV cache for sequential processing
    - Dynamic batching for throughput
    - Quantization (INT8/INT4)
    - Streaming for real-time
    - CUDA graph optimization
    """

    def __init__(
        self,
        model: nn.Module,
        config: InferenceConfig,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(device)
        self.model.eval()

        # Apply optimizations
        self._apply_optimizations()

        # Components
        self.streaming = StreamingInference(model, config, device)
        self.batcher = DynamicBatcher(model, config, device)

        # CUDA graphs
        self.cuda_graphs: Dict[Tuple[int, ...], Any] = {}

        # Stats
        self.total_inferences = 0
        self.total_time = 0.0

    def _apply_optimizations(self):
        """Apply model optimizations."""
        # Quantization
        if self.config.quantization:
            self._quantize_model()

        # Compile with torch.compile if available
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(
                    self.model,
                    mode='reduce-overhead',
                    fullgraph=False
                )
            except Exception:
                pass  # Compilation not available or failed

    def _quantize_model(self):
        """Quantize model weights."""
        bits = 8 if self.config.quantization == 'int8' else 4

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace with quantized version
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                parent = self.model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)

                quant_linear = QuantizedLinear.from_float(module, bits)
                setattr(parent, child_name, quant_linear)

    def warmup(self, input_shape: Tuple[int, ...], num_steps: int = 10):
        """Warmup the model and optionally capture CUDA graphs."""
        dummy_input = torch.randn(*input_shape, device=self.device)

        # Warmup runs
        for _ in range(num_steps):
            with torch.no_grad():
                with autocast(dtype=torch.float16):
                    _ = self.model(dummy_input)

        # Capture CUDA graph if enabled
        if self.config.use_cuda_graphs and self.device == 'cuda':
            self._capture_cuda_graph(input_shape)

    def _capture_cuda_graph(self, input_shape: Tuple[int, ...]):
        """Capture a CUDA graph for the given input shape."""
        static_input = torch.randn(*input_shape, device=self.device)
        static_output = None

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for _ in range(self.config.cudagraph_warmup_steps):
                with torch.no_grad():
                    with autocast(dtype=torch.float16):
                        static_output = self.model(static_input)

        torch.cuda.current_stream().wait_stream(s)

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                with autocast(dtype=torch.float16):
                    static_output = self.model(static_input)

        self.cuda_graphs[input_shape] = (g, static_input, static_output)

    def infer(
        self,
        bci_data: torch.Tensor,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference on BCI data.

        Args:
            bci_data: Input tensor [batch, channels, samples]
            use_cache: Whether to use KV cache

        Returns:
            Dictionary of output tensors
        """
        start_time = time.time()

        bci_data = bci_data.to(self.device)
        input_shape = tuple(bci_data.shape)

        # Try CUDA graph
        if input_shape in self.cuda_graphs:
            g, static_input, static_output = self.cuda_graphs[input_shape]
            static_input.copy_(bci_data)
            g.replay()
            output = {k: v.clone() for k, v in static_output.items()}
        else:
            # Regular inference
            with torch.no_grad():
                with autocast(dtype=torch.float16):
                    output = self.model(bci_data)

        # Update stats
        self.total_inferences += 1
        self.total_time += time.time() - start_time

        return output

    def infer_stream(
        self,
        bci_stream: torch.Tensor,
        chunk_size: Optional[int] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run streaming inference on continuous BCI data.

        Args:
            bci_stream: Continuous input [batch, channels, total_samples]
            chunk_size: Size of each chunk (default from config)

        Returns:
            List of outputs for each chunk
        """
        chunk_size = chunk_size or self.config.stream_chunk_size
        batch_size = bci_stream.shape[0]
        total_samples = bci_stream.shape[-1]

        # Initialize cache
        if hasattr(self.model, 'config'):
            model_config = self.model.config
            num_layers = model_config.num_layers
            num_heads = model_config.num_heads
            head_dim = model_config.embed_dim // model_config.num_heads
        else:
            num_layers, num_heads, head_dim = 32, 32, 128

        kv_cache = KVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_length=self.config.max_cache_length,
            dtype=self.config.cache_dtype,
            device=self.device
        )
        kv_cache.initialize(batch_size)

        outputs = []

        # Process chunks
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            chunk = bci_stream[:, :, start:end].to(self.device)

            with torch.no_grad():
                with autocast(dtype=torch.float16):
                    output = self.model(chunk, kv_cache=kv_cache, use_cache=True)

            outputs.append(output)

        return outputs

    def start_dynamic_batching(self):
        """Start dynamic batching server."""
        self.batcher.start()

    def stop_dynamic_batching(self):
        """Stop dynamic batching server."""
        self.batcher.stop()

    def submit_batch_request(self, bci_data: torch.Tensor) -> int:
        """Submit a request for batched processing."""
        return self.batcher.submit(bci_data)

    def get_batch_result(
        self,
        request_id: int,
        timeout: float = 10.0
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Get result of a batched request."""
        return self.batcher.get_result(request_id, timeout)

    def get_stats(self) -> Dict[str, float]:
        """Get inference statistics."""
        avg_time = self.total_time / max(1, self.total_inferences)
        return {
            'total_inferences': self.total_inferences,
            'total_time_s': self.total_time,
            'avg_latency_ms': avg_time * 1000,
            'throughput_per_s': self.total_inferences / max(0.001, self.total_time)
        }


class SpeculativeDecoder:
    """
    Speculative decoding for faster neural pattern generation.

    Uses a smaller draft model to propose patterns, then
    verifies with the main model. Adapted for BCI signals.
    """

    def __init__(
        self,
        main_model: nn.Module,
        draft_model: nn.Module,
        num_speculative: int = 4,
        device: str = 'cuda'
    ):
        self.main_model = main_model
        self.draft_model = draft_model
        self.num_speculative = num_speculative
        self.device = device

        self.main_model.to(device).eval()
        self.draft_model.to(device).eval()

        # Acceptance stats
        self.total_proposed = 0
        self.total_accepted = 0

    def generate(
        self,
        bci_context: torch.Tensor,
        num_steps: int = 100
    ) -> torch.Tensor:
        """
        Generate neural patterns using speculative decoding.

        Args:
            bci_context: Initial context [batch, channels, samples]
            num_steps: Number of generation steps

        Returns:
            Generated patterns
        """
        current = bci_context.to(self.device)
        generated = []

        steps_done = 0
        while steps_done < num_steps:
            # Draft model proposes
            draft_outputs = []
            draft_current = current

            with torch.no_grad():
                for _ in range(self.num_speculative):
                    with autocast(dtype=torch.float16):
                        draft_out = self.draft_model(draft_current)

                    # Extract pattern
                    pattern = draft_out.get('neural_pattern', draft_current)
                    draft_outputs.append(pattern)
                    draft_current = pattern

            # Main model verifies
            with torch.no_grad():
                with autocast(dtype=torch.float16):
                    # Process all at once for efficiency
                    stacked = torch.cat([current] + draft_outputs, dim=-1)
                    main_out = self.main_model(stacked)

            # Accept/reject (simplified - in practice would compare distributions)
            # Here we accept all drafts if coherence is high
            coherence = main_out.get('coherence', torch.tensor([0.5]))
            accept_prob = coherence.mean().item()

            num_accept = int(self.num_speculative * accept_prob)
            num_accept = max(1, min(num_accept, self.num_speculative))

            # Keep accepted patterns
            for i in range(num_accept):
                generated.append(draft_outputs[i])
                steps_done += 1
                if steps_done >= num_steps:
                    break

            # Update stats
            self.total_proposed += self.num_speculative
            self.total_accepted += num_accept

            # Update context
            if generated:
                current = generated[-1]

        # Concatenate all generated patterns
        if generated:
            return torch.cat(generated, dim=-1)
        return bci_context

    def get_acceptance_rate(self) -> float:
        """Get the speculative acceptance rate."""
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed


def create_inference_engine(
    model: nn.Module,
    quantization: Optional[str] = None,
    use_cuda_graphs: bool = False,
    device: str = 'cuda'
) -> OptimizedInferenceEngine:
    """
    Create an optimized inference engine.

    Args:
        model: The model to optimize
        quantization: 'int8', 'int4', or None
        use_cuda_graphs: Whether to use CUDA graphs
        device: Device to run on

    Returns:
        Configured inference engine
    """
    config = InferenceConfig(
        quantization=quantization,
        use_cuda_graphs=use_cuda_graphs
    )

    return OptimizedInferenceEngine(model, config, device)


# Example usage and testing
if __name__ == "__main__":
    print("Ghost BCI Optimized Inference Engine")
    print("=" * 50)

    # Create a simple test model
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(128, 256, 3, padding=1),
                nn.ReLU(),
            )
            self.fc = nn.Linear(256, 64)

        def forward(self, x, kv_cache=None, use_cache=False):
            encoded = self.encoder(x)
            pooled = encoded.mean(dim=-1)
            output = self.fc(pooled)
            return {
                'neural_pattern': output,
                'coherence': torch.sigmoid(output.mean(dim=-1, keepdim=True))
            }

    # Test on CPU if CUDA not available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model and engine
    model = SimpleTestModel()
    config = InferenceConfig(
        use_cuda_graphs=False,  # Disable for CPU testing
        quantization=None
    )
    engine = OptimizedInferenceEngine(model, config, device)

    # Warmup
    print("\nWarming up...")
    engine.warmup((1, 64, 256), num_steps=5)

    # Test inference
    print("\nTesting inference...")
    test_input = torch.randn(1, 64, 256)
    output = engine.infer(test_input)
    print(f"Output shape: {output['neural_pattern'].shape}")
    print(f"Coherence: {output['coherence'].item():.3f}")

    # Test streaming
    print("\nTesting streaming inference...")
    stream_input = torch.randn(1, 64, 1024)
    outputs = engine.infer_stream(stream_input, chunk_size=256)
    print(f"Processed {len(outputs)} chunks")

    # Show stats
    stats = engine.get_stats()
    print(f"\nStats:")
    print(f"  Total inferences: {stats['total_inferences']}")
    print(f"  Avg latency: {stats['avg_latency_ms']:.2f} ms")
    print(f"  Throughput: {stats['throughput_per_s']:.1f} /s")

    print("\nOptimized inference engine ready!")
