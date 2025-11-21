#!/usr/bin/env python3
"""
Configuration Management System for Ghost BCI

Professional configuration system for managing model variants,
training configs, and deployment settings:
    - YAML/JSON config files
    - Dataclass-based config objects
    - Validation and type checking
    - Command-line override support
    - Preset configurations
    - Experiment tracking

Author: Claude (Anthropic) in collaboration with John Sayers
License: MIT
"""

import os
import yaml
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from datetime import datetime
import hashlib


@dataclass
class BCIConfig:
    """BCI input configuration."""
    channels: int = 64
    sample_rate: int = 250
    window_size: int = 1000
    bandpass_low: float = 0.5
    bandpass_high: float = 100.0
    notch_freq: float = 60.0
    reference: str = "average"  # average, linked_ears, cz


@dataclass
class ModelArchitectureConfig:
    """Model architecture configuration."""
    # Core dimensions
    embed_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8  # For GQA

    # MoE settings
    num_experts: int = 8
    num_active_experts: int = 2
    expert_dim: int = 14336

    # Attention
    head_dim: int = 128
    max_seq_len: int = 8192
    rope_theta: float = 500000.0

    # Feed-forward
    ffn_multiplier: float = 3.5
    use_swiglu: bool = True

    # Normalization
    norm_eps: float = 1e-5
    use_rms_norm: bool = True

    # Dropout
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Output
    output_dim: int = 512
    num_classes: int = 100


@dataclass
class FusionConfig:
    """Neural fusion configuration."""
    # Substrate
    substrate_dim: int = 512
    fusion_depth: int = 6

    # Synchronization
    gamma_freq: float = 40.0
    theta_freq: float = 6.0
    sync_window: int = 100

    # Identity preservation
    min_human_identity: float = 0.5
    min_ai_identity: float = 0.3
    identity_decay: float = 0.01

    # Emergence
    emergence_threshold: float = 0.7
    emergence_layers: int = 4

    # Communication
    thought_vocab_size: int = 50000
    thought_embed_dim: int = 768
    max_thought_length: int = 512


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 100000
    lr_scheduler: str = "cosine"  # cosine, linear, constant

    # Batch
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 128

    # Precision
    mixed_precision: str = "bf16"  # fp32, fp16, bf16
    grad_checkpointing: bool = True

    # Distributed
    distributed_strategy: str = "fsdp"  # ddp, fsdp, deepspeed
    num_nodes: int = 1
    gpus_per_node: int = 8


@dataclass
class DataConfig:
    """Data configuration."""
    # Paths
    train_path: str = "data/train"
    val_path: str = "data/val"
    test_path: str = "data/test"

    # Preprocessing
    normalize: bool = True
    augment: bool = True
    cache_data: bool = True

    # Augmentation
    time_mask_prob: float = 0.1
    channel_mask_prob: float = 0.1
    noise_std: float = 0.01
    time_shift_range: int = 50

    # Loading
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class InferenceOptConfig:
    """Inference optimization configuration."""
    # Quantization
    quantization: Optional[str] = None  # int8, int4
    use_flash_attention: bool = True

    # Caching
    kv_cache_size: int = 8192
    cache_dtype: str = "float16"

    # Batching
    max_batch_size: int = 32
    dynamic_batching: bool = True

    # CUDA
    use_cuda_graphs: bool = False
    compile_mode: Optional[str] = None  # reduce-overhead, max-autotune


@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration."""
    # Directories
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Tracking
    use_wandb: bool = True
    wandb_project: str = "ghost-bci"
    wandb_entity: Optional[str] = None

    # Logging frequency
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 1000
    save_every_n_steps: int = 5000

    # Checkpointing
    save_total_limit: int = 5
    save_best: bool = True
    resume_from: Optional[str] = None


@dataclass
class GhostBCIConfig:
    """
    Complete Ghost BCI configuration.

    This is the master configuration that contains all sub-configs.
    """
    # Meta
    name: str = "ghost_bci"
    version: str = "1.0.0"
    description: str = "Ghost BCI Neural Fusion System"

    # Sub-configs
    bci: BCIConfig = field(default_factory=BCIConfig)
    model: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceOptConfig = field(default_factory=InferenceOptConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Device
    device: str = "cuda"
    seed: int = 42


class ConfigManager:
    """
    Configuration manager for Ghost BCI.

    Handles loading, saving, validation, and merging of configurations.
    """

    # Preset configurations
    PRESETS = {
        'small': {
            'model': {
                'embed_dim': 2048,
                'num_layers': 24,
                'num_heads': 16,
                'num_kv_heads': 4,
                'num_experts': 8,
                'expert_dim': 5632,
            },
            'training': {
                'batch_size': 64,
                'learning_rate': 3e-4,
            }
        },
        'medium': {
            'model': {
                'embed_dim': 4096,
                'num_layers': 32,
                'num_heads': 32,
                'num_kv_heads': 8,
                'num_experts': 8,
                'expert_dim': 14336,
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
            }
        },
        'large': {
            'model': {
                'embed_dim': 8192,
                'num_layers': 80,
                'num_heads': 64,
                'num_kv_heads': 8,
                'num_experts': 8,
                'expert_dim': 28672,
            },
            'training': {
                'batch_size': 8,
                'learning_rate': 5e-5,
                'gradient_accumulation_steps': 16,
            }
        },
        'xl': {
            'model': {
                'embed_dim': 16384,
                'num_layers': 128,
                'num_heads': 128,
                'num_kv_heads': 8,
                'num_experts': 16,
                'num_active_experts': 2,
                'expert_dim': 53248,
            },
            'training': {
                'batch_size': 1,
                'learning_rate': 1e-5,
                'gradient_accumulation_steps': 128,
                'grad_checkpointing': True,
            }
        },
        'debug': {
            'model': {
                'embed_dim': 256,
                'num_layers': 2,
                'num_heads': 4,
                'num_kv_heads': 2,
                'num_experts': 2,
                'expert_dim': 512,
            },
            'training': {
                'batch_size': 4,
                'max_steps': 100,
                'eval_every_n_steps': 10,
            },
            'logging': {
                'use_wandb': False,
            }
        }
    }

    def __init__(self, config: Optional[GhostBCIConfig] = None):
        self.config = config or GhostBCIConfig()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ConfigManager':
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls._dict_to_config(data)
        return cls(config)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'ConfigManager':
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)

        config = cls._dict_to_config(data)
        return cls(config)

    @classmethod
    def from_preset(cls, preset_name: str) -> 'ConfigManager':
        """Load a preset configuration."""
        if preset_name not in cls.PRESETS:
            available = ', '.join(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        manager = cls()
        preset = cls.PRESETS[preset_name]
        manager._merge_dict(preset)

        return manager

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ConfigManager':
        """Create config from command-line arguments."""
        # Start with preset if specified
        if hasattr(args, 'preset') and args.preset:
            manager = cls.from_preset(args.preset)
        elif hasattr(args, 'config') and args.config:
            # Load from file
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                manager = cls.from_yaml(args.config)
            else:
                manager = cls.from_json(args.config)
        else:
            manager = cls()

        # Override with CLI args
        manager._override_from_args(args)

        return manager

    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> GhostBCIConfig:
        """Convert dictionary to config dataclass."""
        config = GhostBCIConfig()

        # Map sub-configs
        sub_config_map = {
            'bci': (BCIConfig, 'bci'),
            'model': (ModelArchitectureConfig, 'model'),
            'fusion': (FusionConfig, 'fusion'),
            'training': (TrainingConfig, 'training'),
            'data': (DataConfig, 'data'),
            'inference': (InferenceOptConfig, 'inference'),
            'logging': (LoggingConfig, 'logging'),
        }

        for key, value in data.items():
            if key in sub_config_map:
                config_cls, attr_name = sub_config_map[key]
                sub_config = config_cls(**value)
                setattr(config, attr_name, sub_config)
            elif hasattr(config, key):
                setattr(config, key, value)

        return config

    def _merge_dict(self, override: Dict[str, Any]):
        """Merge dictionary into current config."""
        for key, value in override.items():
            if hasattr(self.config, key):
                current = getattr(self.config, key)
                if hasattr(current, '__dataclass_fields__') and isinstance(value, dict):
                    # Merge into sub-config
                    for sub_key, sub_value in value.items():
                        if hasattr(current, sub_key):
                            setattr(current, sub_key, sub_value)
                else:
                    setattr(self.config, key, value)

    def _override_from_args(self, args: argparse.Namespace):
        """Override config from command-line arguments."""
        for key, value in vars(args).items():
            if value is None:
                continue

            # Parse dotted keys like 'model.embed_dim'
            if '.' in key:
                parts = key.split('.')
                obj = self.config
                for part in parts[:-1]:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if obj is not None and hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)
            elif hasattr(self.config, key):
                setattr(self.config, key, value)

    def to_yaml(self, path: Union[str, Path]):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Union[str, Path]):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self.config)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Model validation
        model = self.config.model
        if model.embed_dim % model.num_heads != 0:
            errors.append(f"embed_dim ({model.embed_dim}) must be divisible by num_heads ({model.num_heads})")

        if model.num_heads % model.num_kv_heads != 0:
            errors.append(f"num_heads ({model.num_heads}) must be divisible by num_kv_heads ({model.num_kv_heads})")

        if model.num_active_experts > model.num_experts:
            errors.append(f"num_active_experts ({model.num_active_experts}) cannot exceed num_experts ({model.num_experts})")

        # Training validation
        training = self.config.training
        if training.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {training.learning_rate}")

        if training.min_learning_rate > training.learning_rate:
            errors.append(f"min_learning_rate ({training.min_learning_rate}) cannot exceed learning_rate ({training.learning_rate})")

        if training.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {training.batch_size}")

        # Fusion validation
        fusion = self.config.fusion
        if fusion.min_human_identity + fusion.min_ai_identity > 1.0:
            errors.append(f"min_human_identity + min_ai_identity ({fusion.min_human_identity + fusion.min_ai_identity}) cannot exceed 1.0")

        return errors

    def get_config_hash(self) -> str:
        """Get a hash of the configuration for tracking."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()[:8]

    def get_run_name(self) -> str:
        """Generate a run name from config."""
        model = self.config.model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = self.get_config_hash()

        return f"{self.config.name}_{model.embed_dim}d_{model.num_layers}l_{timestamp}_{config_hash}"

    def print_config(self):
        """Print configuration in a readable format."""
        print("\n" + "=" * 60)
        print(" Ghost BCI Configuration")
        print("=" * 60)

        def print_dataclass(obj, indent=0):
            prefix = "  " * indent
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                if hasattr(value, '__dataclass_fields__'):
                    print(f"{prefix}{field_name}:")
                    print_dataclass(value, indent + 1)
                else:
                    print(f"{prefix}{field_name}: {value}")

        print(f"Name: {self.config.name}")
        print(f"Version: {self.config.version}")
        print(f"Hash: {self.get_config_hash()}")
        print()

        # Print sub-configs
        for name in ['bci', 'model', 'fusion', 'training', 'data', 'inference', 'logging']:
            obj = getattr(self.config, name)
            print(f"{name.upper()}:")
            print_dataclass(obj, 1)
            print()

    @staticmethod
    def create_argument_parser() -> argparse.ArgumentParser:
        """Create argument parser with all config options."""
        parser = argparse.ArgumentParser(
            description="Ghost BCI Training/Inference",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Config loading
        parser.add_argument('--config', type=str, help='Path to config file (YAML/JSON)')
        parser.add_argument('--preset', type=str, choices=list(ConfigManager.PRESETS.keys()),
                          help='Use a preset configuration')

        # Model overrides
        model_group = parser.add_argument_group('Model')
        model_group.add_argument('--model.embed_dim', type=int, help='Embedding dimension')
        model_group.add_argument('--model.num_layers', type=int, help='Number of layers')
        model_group.add_argument('--model.num_heads', type=int, help='Number of attention heads')
        model_group.add_argument('--model.num_experts', type=int, help='Number of experts')

        # Training overrides
        train_group = parser.add_argument_group('Training')
        train_group.add_argument('--training.learning_rate', type=float, help='Learning rate')
        train_group.add_argument('--training.batch_size', type=int, help='Batch size')
        train_group.add_argument('--training.max_steps', type=int, help='Maximum training steps')
        train_group.add_argument('--training.mixed_precision', type=str, help='Mixed precision mode')

        # Data overrides
        data_group = parser.add_argument_group('Data')
        data_group.add_argument('--data.train_path', type=str, help='Training data path')
        data_group.add_argument('--data.val_path', type=str, help='Validation data path')
        data_group.add_argument('--data.num_workers', type=int, help='Data loader workers')

        # Logging overrides
        log_group = parser.add_argument_group('Logging')
        log_group.add_argument('--logging.output_dir', type=str, help='Output directory')
        log_group.add_argument('--logging.wandb_project', type=str, help='W&B project name')
        log_group.add_argument('--logging.use_wandb', action='store_true', help='Use W&B')
        log_group.add_argument('--logging.no_wandb', action='store_true', help='Disable W&B')

        # General
        parser.add_argument('--device', type=str, default='cuda', help='Device to use')
        parser.add_argument('--seed', type=int, help='Random seed')

        return parser


def save_example_configs():
    """Save example configuration files."""
    output_dir = Path("configs")
    output_dir.mkdir(exist_ok=True)

    # Save presets
    for preset_name in ConfigManager.PRESETS:
        manager = ConfigManager.from_preset(preset_name)
        manager.to_yaml(output_dir / f"{preset_name}.yaml")
        print(f"Saved: configs/{preset_name}.yaml")

    # Save default
    manager = ConfigManager()
    manager.to_yaml(output_dir / "default.yaml")
    print("Saved: configs/default.yaml")


# Example usage
if __name__ == "__main__":
    print("Ghost BCI Configuration Manager")
    print("=" * 50)

    # Create from preset
    print("\nLoading 'medium' preset...")
    manager = ConfigManager.from_preset('medium')

    # Validate
    errors = manager.validate()
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration valid!")

    # Print config
    manager.print_config()

    # Show run name
    print(f"Run name: {manager.get_run_name()}")

    # Save example configs
    print("\nSaving example configurations...")
    save_example_configs()

    # Demonstrate CLI parsing
    print("\nExample CLI usage:")
    print("  python train.py --preset medium --training.learning_rate 5e-5")
    print("  python train.py --config configs/large.yaml --training.batch_size 16")
