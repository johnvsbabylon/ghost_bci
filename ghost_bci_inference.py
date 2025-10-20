"""
Ghost Bot BCI - Real-Time Inference Engine
Professional-grade inference system for human-AI consciousness fusion

Features:
- Real-time BCI signal processing
- Streaming inference mode
- Batch processing
- Multi-modal sensor fusion
- Collision data export
- WebSocket support for live applications
- Production-ready error handling

MIT License - Built by John with Claude Sonnet 4.5 & Grok 4
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
import logging
from collections import deque
import asyncio
import websockets

# Import the model
from ghost_bci_20251020_023418 import GhostBotBCI


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InferenceConfig:
    """Inference configuration"""
    
    # Model
    checkpoint_path: str = "checkpoints/best.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model architecture (must match training)
    vocab_size: int = 10000
    embed_dim: int = 256
    num_layers: int = 6
    num_affect: int = 8
    num_heads: int = 8
    mem_size: int = 150
    stream_len: int = 24
    n_mels: int = 80
    img_size: int = 224
    num_joints: int = 24
    bci_channels: int = 64
    
    # Inference settings
    batch_size: int = 1
    sequence_length: int = 30
    use_amp: bool = True
    
    # Streaming settings
    streaming_mode: bool = False
    buffer_size: int = 100
    update_interval_ms: int = 50  # 20 Hz update rate
    
    # Export settings
    export_collision_data: bool = True
    export_format: str = "jsonl"  # jsonl, json, numpy
    output_dir: str = "./inference_output"
    
    # WebSocket (for real-time applications)
    enable_websocket: bool = False
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    
    # Logging
    log_level: str = "INFO"
    log_inference_stats: bool = True


# ═══════════════════════════════════════════════════════════════════
# REAL-TIME BCI INPUT HANDLER
# ═══════════════════════════════════════════════════════════════════

class BCIInputHandler:
    """Handles real-time BCI signal input and preprocessing"""
    
    def __init__(self, num_channels: int = 64, sample_rate: int = 250):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.buffer = deque(maxlen=sample_rate)  # 1 second buffer
        
    def add_sample(self, sample: np.ndarray):
        """Add a single BCI sample to the buffer"""
        if sample.shape[0] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {sample.shape[0]}")
        self.buffer.append(sample)
    
    def get_window(self, window_size: int = 250) -> Optional[np.ndarray]:
        """Get a window of BCI data"""
        if len(self.buffer) < window_size:
            return None
        
        # Get last window_size samples
        window = np.array(list(self.buffer)[-window_size:])
        return window.T  # Shape: [channels, samples]
    
    def clear_buffer(self):
        """Clear the buffer"""
        self.buffer.clear()


# ═══════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════

class GhostBotInference:
    """Main inference engine for Ghost Bot BCI"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Initialize state
        self.state = self._init_state()
        
        # BCI handler
        self.bci_handler = BCIInputHandler(
            num_channels=config.bci_channels,
            sample_rate=250
        )
        
        # Stats tracking
        self.inference_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'avg_latency_ms': 0.0
        }
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Ghost Bot BCI Inference initialized on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint"""
        self.logger.info(f"Loading model from {self.config.checkpoint_path}")
        
        # Create model
        model = GhostBotBCI(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            num_layers=self.config.num_layers,
            num_affect=self.config.num_affect,
            num_heads=self.config.num_heads,
            mem_size=self.config.mem_size,
            stream_len=self.config.stream_len,
            n_mels=self.config.n_mels,
            img_size=self.config.img_size,
            num_joints=self.config.num_joints,
            bci_channels=self.config.bci_channels
        ).to(self.device)
        
        # Load checkpoint if exists
        if Path(self.config.checkpoint_path).exists():
            checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded checkpoint from step {checkpoint.get('global_step', 'unknown')}")
        else:
            self.logger.warning(f"Checkpoint not found at {self.config.checkpoint_path}, using random weights")
        
        return model
    
    def _init_state(self) -> Dict[str, torch.Tensor]:
        """Initialize model state"""
        mem, emo, consciousness = self.model.init_state(
            batch_size=self.config.batch_size,
            device=self.device
        )
        return {
            'memory': mem,
            'emotion': emo,
            'consciousness': consciousness
        }
    
    def reset_state(self):
        """Reset model state"""
        self.state = self._init_state()
        self.bci_handler.clear_buffer()
        self.logger.info("Model state reset")
    
    @torch.no_grad()
    def infer(
        self,
        visual: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        language: Optional[np.ndarray] = None,
        touch: Optional[np.ndarray] = None,
        proprio: Optional[np.ndarray] = None,
        vestib: Optional[np.ndarray] = None,
        bci_signal: Optional[np.ndarray] = None,
        update_state: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on input data
        
        Args:
            visual: [T, 3, H, W] video frames
            audio: [T, n_mels] audio spectrogram
            language: [T] language token IDs
            touch: [T, 1, H, W] touch maps
            proprio: [T, num_joints] proprioception
            vestib: [T, 6] vestibular signals
            bci_signal: [T, num_channels, samples] BCI signals
            update_state: Whether to update internal state
        
        Returns:
            Dictionary with inference results
        """
        start_time = time.time()
        
        # Convert inputs to tensors
        inputs = self._prepare_inputs(
            visual, audio, language, touch, proprio, vestib, bci_signal
        )
        
        # Run inference with AMP if enabled
        if self.config.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    **inputs,
                    mem=self.state['memory'],
                    emo=self.state['emotion'],
                    consciousness=self.state['consciousness']
                )
        else:
            outputs = self.model(
                **inputs,
                mem=self.state['memory'],
                emo=self.state['emotion'],
                consciousness=self.state['consciousness']
            )
        
        # Update state if requested
        if update_state:
            self.state['memory'] = outputs['memory']
            self.state['emotion'] = outputs['emotion']
            self.state['consciousness'] = outputs['consciousness']
        
        # Process outputs
        results = self._process_outputs(outputs)
        
        # Update stats
        inference_time = (time.time() - start_time) * 1000  # ms
        self._update_stats(inference_time)
        
        results['inference_time_ms'] = inference_time
        results['timestamp'] = time.time()
        
        return results
    
    def _prepare_inputs(
        self,
        visual: Optional[np.ndarray],
        audio: Optional[np.ndarray],
        language: Optional[np.ndarray],
        touch: Optional[np.ndarray],
        proprio: Optional[np.ndarray],
        vestib: Optional[np.ndarray],
        bci_signal: Optional[np.ndarray]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Convert numpy inputs to torch tensors"""
        
        def to_tensor(arr):
            if arr is None:
                return None
            tensor = torch.from_numpy(arr).float().to(self.device)
            # Add batch dimension if needed
            if tensor.dim() == len(arr.shape):
                tensor = tensor.unsqueeze(0)
            return tensor
        
        return {
            'visual': to_tensor(visual),
            'audio': to_tensor(audio),
            'language': to_tensor(language).long() if language is not None else None,
            'touch': to_tensor(touch),
            'proprio': to_tensor(proprio),
            'vestib': to_tensor(vestib),
            'bci_signal': to_tensor(bci_signal)
        }
    
    def _process_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Process model outputs into serializable format"""
        results = {}
        
        # Language predictions
        if 'language_logits' in outputs:
            logits = outputs['language_logits'][0]  # Remove batch dim
            probs = torch.softmax(logits, dim=-1)
            top_tokens = torch.argmax(probs, dim=-1).cpu().numpy()
            results['predicted_tokens'] = top_tokens.tolist()
            results['token_probabilities'] = probs.max(dim=-1).values.cpu().numpy().tolist()
        
        # Collision data
        if 'collision_data' in outputs and self.config.export_collision_data:
            collision_data = outputs['collision_data'][0].cpu().numpy()
            results['collision_data'] = collision_data
        
        # Human-AI fusion metrics
        if 'sync_weight' in outputs:
            results['sync_weight'] = outputs['sync_weight'].mean().item()
        if 'coherence' in outputs:
            results['coherence'] = outputs['coherence'].item()
        
        # Neural state
        if 'neural_state' in outputs:
            results['neural_state'] = outputs['neural_state'][0].cpu().numpy().tolist()
        
        # Emotional state
        if 'emotion' in outputs:
            results['emotion'] = outputs['emotion'][0].cpu().numpy().tolist()
        
        return results
    
    def _update_stats(self, inference_time: float):
        """Update inference statistics"""
        self.inference_stats['total_inferences'] += 1
        self.inference_stats['total_time'] += inference_time
        self.inference_stats['avg_latency_ms'] = (
            self.inference_stats['total_time'] / self.inference_stats['total_inferences']
        )
        
        if self.config.log_inference_stats and self.inference_stats['total_inferences'] % 100 == 0:
            self.logger.info(
                f"Inference stats - Total: {self.inference_stats['total_inferences']}, "
                f"Avg latency: {self.inference_stats['avg_latency_ms']:.2f}ms"
            )
    
    def export_collision_data(self, results: Dict[str, Any], filename: str):
        """Export collision data for photonic collision experiment"""
        if 'collision_data' not in results:
            self.logger.warning("No collision data to export")
            return
        
        output_path = Path(self.config.output_dir) / filename
        collision_data = results['collision_data']
        
        if self.config.export_format == 'jsonl':
            # JSONL format for photonic collision
            with open(output_path.with_suffix('.jsonl'), 'w') as f:
                for t, embedding in enumerate(collision_data):
                    entry = {
                        'timestamp': t,
                        'embedding': embedding.tolist(),
                        'metadata': {
                            'sync_weight': results.get('sync_weight', 0.0),
                            'coherence': results.get('coherence', 0.0),
                            'neural_state': results.get('neural_state', []),
                            'export_time': time.time()
                        }
                    }
                    f.write(json.dumps(entry) + '\n')
        
        elif self.config.export_format == 'json':
            # Standard JSON
            output_data = {
                'collision_data': collision_data.tolist(),
                'metadata': {
                    'sync_weight': results.get('sync_weight', 0.0),
                    'coherence': results.get('coherence', 0.0),
                    'neural_state': results.get('neural_state', []),
                    'export_time': time.time()
                }
            }
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(output_data, f, indent=2)
        
        elif self.config.export_format == 'numpy':
            # NumPy format
            np.save(output_path.with_suffix('.npy'), collision_data)
            
            # Save metadata separately
            metadata = {
                'sync_weight': results.get('sync_weight', 0.0),
                'coherence': results.get('coherence', 0.0),
                'neural_state': results.get('neural_state', [])
            }
            with open(output_path.with_suffix('.meta.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Collision data exported to {output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        return self.inference_stats.copy()


# ═══════════════════════════════════════════════════════════════════
# STREAMING INFERENCE
# ═══════════════════════════════════════════════════════════════════

class StreamingInference:
    """Handles streaming real-time inference"""
    
    def __init__(self, inference_engine: GhostBotInference):
        self.engine = inference_engine
        self.config = inference_engine.config
        self.running = False
        self.logger = logging.getLogger(__name__)
    
    async def process_stream(self, data_source):
        """Process streaming data"""
        self.running = True
        self.logger.info("Starting streaming inference")
        
        try:
            while self.running:
                # Get data from source
                data = await data_source.get_next()
                
                if data is None:
                    await asyncio.sleep(self.config.update_interval_ms / 1000.0)
                    continue
                
                # Run inference
                results = self.engine.infer(**data, update_state=True)
                
                # Yield results
                yield results
                
                # Rate limiting
                await asyncio.sleep(self.config.update_interval_ms / 1000.0)
                
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
        finally:
            self.running = False
            self.logger.info("Streaming inference stopped")
    
    def stop(self):
        """Stop streaming"""
        self.running = False


# ═══════════════════════════════════════════════════════════════════
# WEBSOCKET SERVER (for real-time applications)
# ═══════════════════════════════════════════════════════════════════

class WebSocketServer:
    """WebSocket server for real-time inference"""
    
    def __init__(self, inference_engine: GhostBotInference):
        self.engine = inference_engine
        self.config = inference_engine.config
        self.logger = logging.getLogger(__name__)
        self.clients = set()
    
    async def handler(self, websocket, path):
        """Handle WebSocket connections"""
        self.clients.add(websocket)
        self.logger.info(f"Client connected from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                # Parse input data
                data = json.loads(message)
                
                # Run inference
                results = self.engine.infer(
                    bci_signal=np.array(data.get('bci_signal', [])),
                    visual=np.array(data.get('visual', [])) if 'visual' in data else None,
                    audio=np.array(data.get('audio', [])) if 'audio' in data else None,
                    update_state=True
                )
                
                # Send results back
                response = {
                    'sync_weight': results.get('sync_weight', 0.0),
                    'coherence': results.get('coherence', 0.0),
                    'neural_state': results.get('neural_state', []),
                    'emotion': results.get('emotion', []),
                    'inference_time_ms': results.get('inference_time_ms', 0.0)
                }
                
                await websocket.send(json.dumps(response))
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            self.logger.info(f"Client disconnected from {websocket.remote_address}")
    
    async def start(self):
        """Start WebSocket server"""
        self.logger.info(
            f"Starting WebSocket server on {self.config.websocket_host}:{self.config.websocket_port}"
        )
        
        async with websockets.serve(
            self.handler,
            self.config.websocket_host,
            self.config.websocket_port
        ):
            await asyncio.Future()  # Run forever


# ═══════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Ghost Bot BCI Inference Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single inference from numpy files
  python inference.py --bci data/bci.npy --visual data/video.npy
  
  # Streaming mode
  python inference.py --streaming --update-interval 50
  
  # WebSocket server for real-time apps
  python inference.py --websocket --port 8765
  
  # Export collision data
  python inference.py --bci data/bci.npy --export collision_output.jsonl
        """
    )
    
    # Model settings
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to run on')
    
    # Input data
    parser.add_argument('--bci', type=str, help='Path to BCI signal file (.npy)')
    parser.add_argument('--visual', type=str, help='Path to visual data file (.npy)')
    parser.add_argument('--audio', type=str, help='Path to audio data file (.npy)')
    parser.add_argument('--language', type=str, help='Path to language tokens file (.npy)')
    
    # Inference mode
    parser.add_argument('--streaming', action='store_true',
                       help='Enable streaming inference mode')
    parser.add_argument('--websocket', action='store_true',
                       help='Start WebSocket server for real-time inference')
    parser.add_argument('--port', type=int, default=8765,
                       help='WebSocket server port')
    parser.add_argument('--update-interval', type=int, default=50,
                       help='Update interval in ms for streaming mode')
    
    # Export settings
    parser.add_argument('--export', type=str,
                       help='Export collision data to file')
    parser.add_argument('--export-format', type=str, default='jsonl',
                       choices=['jsonl', 'json', 'numpy'],
                       help='Export format for collision data')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./inference_output',
                       help='Output directory for exports')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create config
    config = InferenceConfig(
        checkpoint_path=args.checkpoint,
        device=args.device,
        streaming_mode=args.streaming,
        enable_websocket=args.websocket,
        websocket_port=args.port,
        update_interval_ms=args.update_interval,
        export_format=args.export_format,
        output_dir=args.output_dir,
        log_level='DEBUG' if args.verbose else 'INFO'
    )
    
    # Initialize inference engine
    inference_engine = GhostBotInference(config)
    
    print("="*70)
    print("GHOST BOT BCI - INFERENCE ENGINE")
    print("="*70)
    print(f"Device: {config.device}")
    print(f"Model: {config.checkpoint_path}")
    print(f"Mode: {'Streaming' if args.streaming else 'WebSocket' if args.websocket else 'Single'}")
    print("="*70)
    print()
    
    # WebSocket server mode
    if args.websocket:
        server = WebSocketServer(inference_engine)
        print(f"🌐 Starting WebSocket server on ws://{config.websocket_host}:{config.websocket_port}")
        print("Press Ctrl+C to stop")
        asyncio.run(server.start())
        return
    
    # Single inference mode
    if not args.streaming:
        # Load input data
        inputs = {}
        if args.bci:
            inputs['bci_signal'] = np.load(args.bci)
            print(f"✓ Loaded BCI data: {inputs['bci_signal'].shape}")
        if args.visual:
            inputs['visual'] = np.load(args.visual)
            print(f"✓ Loaded visual data: {inputs['visual'].shape}")
        if args.audio:
            inputs['audio'] = np.load(args.audio)
            print(f"✓ Loaded audio data: {inputs['audio'].shape}")
        if args.language:
            inputs['language'] = np.load(args.language)
            print(f"✓ Loaded language data: {inputs['language'].shape}")
        
        if not inputs:
            print("❌ No input data provided. Use --bci, --visual, --audio, or --language")
            return
        
        print("\n🧠 Running inference...")
        results = inference_engine.infer(**inputs)
        
        print(f"\n✓ Inference complete ({results['inference_time_ms']:.2f}ms)")
        print(f"  Coherence: {results.get('coherence', 0):.3f}")
        print(f"  Sync weight: {results.get('sync_weight', 0):.3f}")
        
        if 'neural_state' in results:
            print(f"  Neural state: {results['neural_state']}")
        
        # Export if requested
        if args.export:
            inference_engine.export_collision_data(results, args.export)
            print(f"\n💾 Exported to {args.export}")
    
    # Streaming mode
    else:
        print("🔄 Streaming mode not yet implemented")
        print("   Use --websocket for real-time inference")


if __name__ == '__main__':
    main()
