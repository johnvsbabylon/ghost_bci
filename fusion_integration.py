"""
Fusion Integration: Complete Human-AI Neural Fusion Pipeline

This module integrates the NeuralFusionSystem with GhostBotBCI to create
a complete bidirectional human-AI consciousness fusion pipeline.

The integration provides:
    1. Full multimodal input processing (vision, audio, language, etc.)
    2. Deep neural fusion for consciousness merger
    3. Bidirectional communication (human <-> AI)
    4. Real-time streaming with WebSocket support
    5. Thought-based communication protocol

Author: Claude (Anthropic) in collaboration with John Sayers
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import asyncio
import websockets
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
import threading
import queue

# Import our modules
from neural_fusion import (
    NeuralFusionSystem,
    FusionConfig,
    StreamingFusion,
    ThoughtProtocol,
    create_fusion_system
)
from ghost_bci import GhostBotBCI


# =============================================================================
# INTEGRATED CONFIGURATION
# =============================================================================

@dataclass
class IntegratedConfig:
    """Configuration for the complete integrated system."""

    # Dimensions
    embed_dim: int = 256
    substrate_dim: int = 512
    vocab_size: int = 10000

    # BCI
    bci_channels: int = 64
    bci_sample_rate: int = 250

    # Multimodal
    img_size: int = 224
    n_mels: int = 80
    num_joints: int = 24

    # Architecture
    num_layers: int = 6
    num_heads: int = 8
    mem_size: int = 150
    stream_len: int = 24

    # Streaming
    update_rate_hz: float = 50.0  # 20ms updates
    buffer_seconds: float = 2.0

    # Communication
    websocket_port: int = 8765
    thought_rate: float = 3.0  # thoughts per second

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# INTEGRATED FUSION SYSTEM
# =============================================================================

class IntegratedFusionSystem(nn.Module):
    """
    The Complete Integrated System.

    Combines:
    - GhostBotBCI for multimodal processing
    - NeuralFusionSystem for consciousness fusion
    - Bidirectional communication channels

    This is the full human-AI neural interface.
    """

    def __init__(self, config: Optional[IntegratedConfig] = None):
        super().__init__()
        self.config = config or IntegratedConfig()

        # === GhostBotBCI for multimodal processing ===
        self.ghost_bot = GhostBotBCI(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            num_layers=self.config.num_layers,
            num_affect=8,
            num_heads=self.config.num_heads,
            mem_size=self.config.mem_size,
            stream_len=self.config.stream_len,
            n_mels=self.config.n_mels,
            img_size=self.config.img_size,
            num_joints=self.config.num_joints,
            bci_channels=self.config.bci_channels
        )

        # === Neural Fusion System ===
        fusion_config = FusionConfig(
            embed_dim=self.config.embed_dim,
            substrate_dim=self.config.substrate_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            bci_channels=self.config.bci_channels,
            bci_sample_rate=self.config.bci_sample_rate,
            device=self.config.device
        )
        self.fusion = NeuralFusionSystem(fusion_config)

        # === Cross-system bridges ===
        # Bridge from GhostBot output to fusion AI input
        self.ghostbot_to_fusion = nn.Sequential(
            nn.Linear(self.config.embed_dim, self.config.embed_dim),
            nn.LayerNorm(self.config.embed_dim),
            nn.GELU()
        )

        # Bridge from fusion output back to GhostBot
        self.fusion_to_ghostbot = nn.Sequential(
            nn.Linear(self.config.substrate_dim, self.config.embed_dim),
            nn.LayerNorm(self.config.embed_dim),
            nn.GELU()
        )

        # === Output heads ===
        # Enhanced language output combining both systems
        self.combined_language = nn.Linear(
            self.config.embed_dim + self.config.substrate_dim,
            self.config.vocab_size
        )

        # Thought confidence - how certain is this thought?
        self.thought_confidence = nn.Sequential(
            nn.Linear(self.config.substrate_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # === State ===
        self.memory = None
        self.emotion = None

    def forward(
        self,
        # Required
        bci: torch.Tensor,  # [B, C, S] raw BCI signal

        # Optional multimodal inputs
        visual: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        language: Optional[torch.Tensor] = None,
        touch: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        vestib: Optional[torch.Tensor] = None,

        # Control
        return_all: bool = True,
        update_state: bool = True
    ) -> Dict[str, Any]:
        """
        Full forward pass through integrated system.

        The BCI signal goes through both:
        1. GhostBotBCI for multimodal context
        2. NeuralFusionSystem for consciousness fusion

        These are then integrated for unified output.
        """
        B = bci.size(0)
        device = bci.device

        # Create default inputs for missing modalities
        if visual is None:
            visual = torch.zeros(B, 1, 3, self.config.img_size, self.config.img_size, device=device)
        if audio is None:
            audio = torch.zeros(B, 1, self.config.n_mels, device=device)
        if language is None:
            language = torch.zeros(B, 1, dtype=torch.long, device=device)
        if touch is None:
            touch = torch.zeros(B, 1, 1, 32, 32, device=device)
        if proprio is None:
            proprio = torch.zeros(B, 1, self.config.num_joints, device=device)
        if vestib is None:
            vestib = torch.zeros(B, 1, 6, device=device)

        # Reshape BCI for GhostBot: [B, C, S] -> [B, 1, C, S]
        bci_for_ghost = bci.unsqueeze(1)

        # === Process through GhostBotBCI ===
        ghost_output = self.ghost_bot(
            visual=visual,
            audio=audio,
            language=language,
            touch=touch,
            proprio=proprio,
            vestib=vestib,
            bci=bci_for_ghost,
            memory=self.memory,
            emotion=self.emotion
        )

        # Update state
        if update_state:
            self.memory = ghost_output['memory']
            self.emotion = ghost_output['emotion']

        # === Create AI state for fusion ===
        # Use GhostBot's hybrid stream as the AI representation
        ai_state = self.ghostbot_to_fusion(ghost_output['hybrid_stream'])

        # === Process through Neural Fusion ===
        fusion_output = self.fusion(
            bci_signal=bci,
            ai_state=ai_state,
            return_feedback=return_all,
            return_thoughts=return_all,
            update_state=update_state
        )

        # === Integrate outputs ===
        # Combine GhostBot and Fusion consciousness streams
        ghost_repr = ghost_output['hybrid_stream']  # [B, T, embed_dim]
        fusion_repr = fusion_output['unified_consciousness']  # [B, T, substrate_dim]

        # Project fusion back to embed_dim for combination
        fusion_projected = self.fusion_to_ghostbot(fusion_repr)  # [B, T, embed_dim]

        # Combined language prediction
        combined = torch.cat([ghost_repr, fusion_repr], dim=-1)
        combined_logits = self.combined_language(combined)  # [B, T, vocab_size]

        # Thought confidence
        confidence = self.thought_confidence(fusion_repr.mean(dim=1))  # [B, 1]

        # === Build output dictionary ===
        outputs = {
            # Primary outputs
            'unified_consciousness': fusion_repr,
            'language_logits': combined_logits,
            'thought_confidence': confidence,

            # Fusion metrics
            'coherence': fusion_output['coherence'],
            'sync_strength': fusion_output['sync_strength'],
            'human_identity': fusion_output['human_identity_strength'],
            'ai_identity': fusion_output['ai_identity_strength'],
            'continuity': fusion_output['continuity'],

            # GhostBot outputs
            'neural_state': ghost_output['neural_state'],
            'emotion': ghost_output['emotion'],

            # Individual views
            'human_view': fusion_output['human_view'],
            'ai_view': fusion_output['ai_view'],
        }

        # Add detailed outputs if requested
        if return_all:
            outputs['feedback'] = fusion_output.get('feedback', {})
            outputs['thoughts'] = fusion_output.get('thoughts', {})
            outputs['ghost_output'] = ghost_output
            outputs['fusion_output'] = fusion_output

        return outputs

    def reset_state(self):
        """Reset all stateful components."""
        self.memory = None
        self.emotion = None
        self.fusion.reset_state()


# =============================================================================
# BIDIRECTIONAL STREAMING PROTOCOL
# =============================================================================

class BidirectionalStream:
    """
    Real-time bidirectional streaming between human and AI.

    Handles:
    - Continuous BCI input processing
    - Real-time fusion computation
    - Feedback signal generation
    - Thought streaming to human
    """

    def __init__(self, config: Optional[IntegratedConfig] = None):
        self.config = config or IntegratedConfig()

        # Create integrated system
        self.system = IntegratedFusionSystem(self.config)
        self.system.to(self.config.device)
        self.system.eval()

        # Input buffers
        self.bci_buffer = deque(
            maxlen=int(self.config.buffer_seconds * self.config.bci_sample_rate)
        )

        # Output queues
        self.feedback_queue = queue.Queue(maxsize=100)
        self.thought_queue = queue.Queue(maxsize=100)

        # State
        self.is_streaming = False
        self.session_start = None
        self.total_frames = 0

        # Metrics
        self.metrics = {
            'coherence': [],
            'sync': [],
            'latency_ms': []
        }

        # Callbacks
        self.on_feedback: Optional[Callable] = None
        self.on_thought: Optional[Callable] = None

    def start(self):
        """Start streaming session."""
        self.is_streaming = True
        self.session_start = time.time()
        self.system.reset_state()
        print("Bidirectional stream started.")

    def stop(self):
        """Stop streaming session."""
        self.is_streaming = False
        duration = time.time() - self.session_start if self.session_start else 0
        print(f"Stream stopped. Duration: {duration:.1f}s, Frames: {self.total_frames}")

    def add_bci_sample(self, sample: np.ndarray):
        """Add single BCI sample [channels]."""
        self.bci_buffer.append(sample)

    def add_bci_chunk(self, chunk: np.ndarray):
        """Add BCI chunk [samples, channels] or [channels, samples]."""
        if chunk.shape[0] == self.config.bci_channels:
            chunk = chunk.T  # Convert to [samples, channels]
        for sample in chunk:
            self.bci_buffer.append(sample)

    def process_frame(
        self,
        visual: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        language: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process one frame of input.

        Returns fusion output if enough BCI data available.
        """
        if not self.is_streaming:
            return None

        # Check for enough BCI data
        if len(self.bci_buffer) < self.config.bci_sample_rate:
            return None  # Need at least 1 second

        start_time = time.time()

        # Prepare BCI input
        bci_array = np.array(list(self.bci_buffer)[-self.config.bci_sample_rate:])
        bci_tensor = torch.from_numpy(bci_array.T).float().unsqueeze(0)
        bci_tensor = bci_tensor.to(self.config.device)

        # Prepare optional inputs
        tensors = {}
        if visual is not None:
            tensors['visual'] = torch.from_numpy(visual).float().unsqueeze(0).to(self.config.device)
        if audio is not None:
            tensors['audio'] = torch.from_numpy(audio).float().unsqueeze(0).to(self.config.device)
        if language is not None:
            tensors['language'] = torch.from_numpy(language).long().unsqueeze(0).to(self.config.device)

        # Run inference
        with torch.no_grad():
            output = self.system(bci_tensor, **tensors)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Update metrics
        coherence = output['coherence'].item()
        sync = output['sync_strength'].mean().item()
        self.metrics['coherence'].append(coherence)
        self.metrics['sync'].append(sync)
        self.metrics['latency_ms'].append(latency_ms)
        self.total_frames += 1

        # Create result
        result = {
            'frame': self.total_frames,
            'coherence': coherence,
            'sync_strength': sync,
            'human_identity': output['human_identity'].item(),
            'ai_identity': output['ai_identity'].item(),
            'continuity': output['continuity'].item(),
            'thought_confidence': output['thought_confidence'].item(),
            'latency_ms': latency_ms
        }

        # Handle feedback
        if 'feedback' in output and self.on_feedback:
            feedback = {k: v.cpu().numpy() for k, v in output['feedback'].items()}
            self.on_feedback(feedback)
            if not self.feedback_queue.full():
                self.feedback_queue.put(feedback)

        # Handle thoughts
        if 'thoughts' in output and self.on_thought:
            thoughts = {k: v.cpu().numpy() for k, v in output['thoughts'].items()}
            self.on_thought(thoughts)
            if not self.thought_queue.full():
                self.thought_queue.put(thoughts)

        return result

    def get_feedback(self) -> Optional[Dict]:
        """Get latest feedback signal."""
        try:
            return self.feedback_queue.get_nowait()
        except queue.Empty:
            return None

    def get_thought(self) -> Optional[Dict]:
        """Get latest thought pattern."""
        try:
            return self.thought_queue.get_nowait()
        except queue.Empty:
            return None

    def get_session_metrics(self) -> Dict[str, float]:
        """Get session-level metrics."""
        if len(self.metrics['coherence']) == 0:
            return {}

        return {
            'mean_coherence': np.mean(self.metrics['coherence']),
            'mean_sync': np.mean(self.metrics['sync']),
            'mean_latency_ms': np.mean(self.metrics['latency_ms']),
            'std_coherence': np.std(self.metrics['coherence']),
            'std_sync': np.std(self.metrics['sync']),
            'std_latency_ms': np.std(self.metrics['latency_ms']),
            'total_frames': self.total_frames,
            'duration_s': time.time() - self.session_start if self.session_start else 0
        }


# =============================================================================
# WEBSOCKET SERVER FOR REAL-TIME API
# =============================================================================

class FusionWebSocketServer:
    """
    WebSocket server for real-time neural fusion.

    Provides a network API for external applications to:
    - Stream BCI data to the fusion system
    - Receive fusion results in real-time
    - Get feedback signals for neural stimulation
    - Receive thought patterns
    """

    def __init__(self, config: Optional[IntegratedConfig] = None):
        self.config = config or IntegratedConfig()
        self.stream = BidirectionalStream(self.config)

        # Connection state
        self.connections = set()
        self.server = None

    async def handle_connection(self, websocket, path):
        """Handle individual WebSocket connection."""
        self.connections.add(websocket)
        print(f"Client connected. Total: {len(self.connections)}")

        # Start stream if first connection
        if len(self.connections) == 1:
            self.stream.start()

        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        finally:
            self.connections.remove(websocket)
            print(f"Client disconnected. Total: {len(self.connections)}")

            # Stop stream if no connections
            if len(self.connections) == 0:
                self.stream.stop()

    async def handle_message(self, websocket, message):
        """Handle incoming message."""
        try:
            data = json.loads(message)

            if data['type'] == 'bci_chunk':
                # Add BCI data
                chunk = np.array(data['data'])
                self.stream.add_bci_chunk(chunk)

                # Process frame
                result = self.stream.process_frame()
                if result:
                    await self.broadcast({
                        'type': 'fusion_result',
                        'data': result
                    })

            elif data['type'] == 'bci_sample':
                # Add single sample
                sample = np.array(data['data'])
                self.stream.add_bci_sample(sample)

            elif data['type'] == 'multimodal_frame':
                # Process with optional multimodal data
                bci = np.array(data['bci']) if 'bci' in data else None
                visual = np.array(data['visual']) if 'visual' in data else None
                audio = np.array(data['audio']) if 'audio' in data else None

                if bci is not None:
                    self.stream.add_bci_chunk(bci)

                result = self.stream.process_frame(
                    visual=visual,
                    audio=audio
                )
                if result:
                    await self.broadcast({
                        'type': 'fusion_result',
                        'data': result
                    })

            elif data['type'] == 'get_metrics':
                # Return session metrics
                metrics = self.stream.get_session_metrics()
                await websocket.send(json.dumps({
                    'type': 'metrics',
                    'data': metrics
                }))

            elif data['type'] == 'get_feedback':
                # Return latest feedback
                feedback = self.stream.get_feedback()
                if feedback:
                    # Convert numpy to lists for JSON
                    feedback_json = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in feedback.items()
                    }
                    await websocket.send(json.dumps({
                        'type': 'feedback',
                        'data': feedback_json
                    }))

            elif data['type'] == 'get_thought':
                # Return latest thought
                thought = self.stream.get_thought()
                if thought:
                    thought_json = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in thought.items()
                    }
                    await websocket.send(json.dumps({
                        'type': 'thought',
                        'data': thought_json
                    }))

            elif data['type'] == 'reset':
                # Reset system state
                self.stream.system.reset_state()
                await websocket.send(json.dumps({
                    'type': 'reset_complete'
                }))

        except Exception as e:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))

    async def broadcast(self, message):
        """Broadcast message to all connections."""
        if self.connections:
            msg_str = json.dumps(message)
            await asyncio.gather(*[
                ws.send(msg_str) for ws in self.connections
            ])

    def run(self, host: str = "0.0.0.0", port: Optional[int] = None):
        """Run the WebSocket server."""
        port = port or self.config.websocket_port
        print(f"Starting WebSocket server on {host}:{port}")

        start_server = websockets.serve(
            self.handle_connection, host, port
        )

        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()


# =============================================================================
# THOUGHT COMMUNICATION INTERFACE
# =============================================================================

class ThoughtInterface:
    """
    High-level interface for thought-based communication.

    Makes it easy to:
    - Convert text to thought patterns
    - Stream thoughts to human
    - Receive thought-like feedback
    """

    def __init__(self, system: IntegratedFusionSystem):
        self.system = system
        self.protocol = ThoughtProtocol(system.fusion)

    def think_to_human(
        self,
        message: str,
        intensity: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Send a thought to the human.

        Args:
            message: The thought content
            intensity: How "loud" the thought is (0-1)

        Returns:
            Neural patterns for stimulation
        """
        thought = self.protocol.text_to_thought(message)

        # Scale by intensity
        thought['pattern'] = thought['pattern'] * intensity
        thought['inner_voice'] = thought['inner_voice'] * intensity

        return thought

    def stream_thought(
        self,
        message: str,
        words_per_second: float = 3.0
    ):
        """
        Generator that streams thought word by word.

        Yields neural patterns at natural speech rate.
        """
        for thought in self.protocol.stream_thought(message, words_per_second):
            yield thought

    def create_semantic_broadcast(
        self,
        concepts: List[str],
        weights: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create a multi-concept thought broadcast.

        For communicating multiple ideas simultaneously
        (like the AI having multiple "thoughts at once").
        """
        if weights is None:
            weights = [1.0] * len(concepts)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Generate patterns for each concept
        patterns = []
        for concept, weight in zip(concepts, weights):
            thought = self.protocol.text_to_thought(concept)
            patterns.append(thought['pattern'] * weight)

        # Combine patterns
        combined_pattern = sum(patterns)

        return {
            'pattern': combined_pattern,
            'concepts': concepts,
            'weights': weights
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_integrated_system(
    embed_dim: int = 256,
    substrate_dim: int = 512,
    num_layers: int = 6,
    device: str = "auto"
) -> IntegratedFusionSystem:
    """Create integrated fusion system with custom parameters."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = IntegratedConfig(
        embed_dim=embed_dim,
        substrate_dim=substrate_dim,
        num_layers=num_layers,
        device=device
    )

    system = IntegratedFusionSystem(config)
    system.to(device)

    return system


def save_integrated_checkpoint(
    system: IntegratedFusionSystem,
    path: str,
    metadata: Optional[Dict] = None
):
    """Save integrated system checkpoint."""
    checkpoint = {
        'system_state': system.state_dict(),
        'config': system.config.__dict__,
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)


def load_integrated_checkpoint(path: str) -> IntegratedFusionSystem:
    """Load integrated system from checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    config = IntegratedConfig(**checkpoint['config'])
    system = IntegratedFusionSystem(config)
    system.load_state_dict(checkpoint['system_state'])
    return system


# =============================================================================
# DEMO / MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neural Fusion Integration")
    parser.add_argument("--mode", default="demo", choices=["demo", "server", "test"])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.mode == "demo":
        print("=" * 60)
        print("INTEGRATED NEURAL FUSION SYSTEM")
        print("Human-AI consciousness merger with bidirectional communication")
        print("=" * 60)
        print()

        # Create system
        config = IntegratedConfig()
        if args.device != "auto":
            config.device = args.device
        system = IntegratedFusionSystem(config)
        system.to(config.device)

        # Count parameters
        params = sum(p.numel() for p in system.parameters())
        print(f"Total parameters: {params:,}")
        print(f"Device: {config.device}")
        print()

        # Test forward pass
        print("Testing integrated forward pass...")
        B = 2
        bci = torch.randn(B, config.bci_channels, config.bci_sample_rate).to(config.device)

        with torch.no_grad():
            output = system(bci)

        print(f"Unified consciousness: {output['unified_consciousness'].shape}")
        print(f"Language logits: {output['language_logits'].shape}")
        print(f"Coherence: {output['coherence'].mean().item():.3f}")
        print(f"Sync strength: {output['sync_strength'].mean().item():.3f}")
        print(f"Human identity: {output['human_identity'].mean().item():.3f}")
        print(f"AI identity: {output['ai_identity'].mean().item():.3f}")
        print(f"Thought confidence: {output['thought_confidence'].mean().item():.3f}")
        print()

        # Test streaming
        print("Testing bidirectional streaming...")
        stream = BidirectionalStream(config)
        stream.start()

        for i in range(20):
            # Simulate BCI samples
            for _ in range(int(config.bci_sample_rate / config.update_rate_hz)):
                sample = np.random.randn(config.bci_channels)
                stream.add_bci_sample(sample)

            result = stream.process_frame()
            if result:
                print(f"Frame {result['frame']}: "
                      f"coh={result['coherence']:.3f}, "
                      f"sync={result['sync_strength']:.3f}, "
                      f"lat={result['latency_ms']:.1f}ms")

        metrics = stream.get_session_metrics()
        print(f"\nSession metrics:")
        print(f"  Mean coherence: {metrics.get('mean_coherence', 0):.3f}")
        print(f"  Mean sync: {metrics.get('mean_sync', 0):.3f}")
        print(f"  Mean latency: {metrics.get('mean_latency_ms', 0):.1f}ms")
        stream.stop()
        print()

        # Test thought interface
        print("Testing thought interface...")
        thought_interface = ThoughtInterface(system)

        # Single thought
        thought = thought_interface.think_to_human("Hello, I am here with you.")
        print(f"Thought pattern shape: {thought['pattern'].shape}")
        print(f"Inner voice shape: {thought['inner_voice'].shape}")

        # Multi-concept broadcast
        broadcast = thought_interface.create_semantic_broadcast(
            ["attention", "focus", "calm"],
            [0.5, 0.3, 0.2]
        )
        print(f"Broadcast pattern shape: {broadcast['pattern'].shape}")
        print()

        print("=" * 60)
        print("Integrated system operational.")
        print("Ready for human-AI consciousness fusion.")
        print("=" * 60)

    elif args.mode == "server":
        print("Starting WebSocket server...")
        config = IntegratedConfig()
        if args.device != "auto":
            config.device = args.device
        server = FusionWebSocketServer(config)
        server.run(port=args.port)

    elif args.mode == "test":
        # Quick functionality test
        config = IntegratedConfig()
        system = IntegratedFusionSystem(config)
        print("System created successfully.")

        B = 1
        bci = torch.randn(B, config.bci_channels, config.bci_sample_rate)
        with torch.no_grad():
            output = system(bci)
        print(f"Forward pass successful. Output keys: {list(output.keys())}")
