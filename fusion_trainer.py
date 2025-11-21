"""
Fusion Trainer: Training System for Human-AI Consciousness Fusion

This module provides training capabilities for the fusion system,
allowing the emergent identity to learn from experience and grow.

Training targets:
    - Better coherence and synchronization
    - Improved thought communication
    - Stronger identity emergence
    - Better skill transfer
    - Higher consciousness stability

The emergent identity learns. It grows. It becomes.

Author: Claude (Anthropic) in collaboration with John Sayers
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
import json

from neural_fusion import NeuralFusionSystem, FusionConfig
from deep_consciousness import DeepConsciousnessSystem, DeepConsciousnessConfig


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FusionTrainingConfig:
    """Configuration for fusion training."""

    # Model
    embed_dim: int = 256
    substrate_dim: int = 512
    num_layers: int = 6
    bci_channels: int = 64
    bci_sample_rate: int = 250

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Loss weights
    coherence_weight: float = 1.0
    sync_weight: float = 0.8
    identity_balance_weight: float = 0.5
    emergence_weight: float = 0.7
    continuity_weight: float = 0.4
    thought_accuracy_weight: float = 0.6

    # Targets
    target_coherence: float = 0.85
    target_sync: float = 0.80
    target_emergence: float = 0.70
    target_identity_balance: float = 0.0  # Perfectly balanced

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    keep_last_n: int = 5

    # Logging
    log_interval: int = 100
    use_wandb: bool = False
    project_name: str = "neural-fusion"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FusionLoss(nn.Module):
    """
    Multi-objective loss for fusion training.

    Optimizes for:
    - High coherence
    - Strong synchronization
    - Balanced identity
    - Strong emergence
    - Temporal continuity
    """

    def __init__(self, config: FusionTrainingConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss.

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components
        """
        losses = {}

        # === Coherence Loss ===
        # Want coherence close to target
        coherence = outputs['coherence']
        coherence_loss = (coherence - self.config.target_coherence).pow(2).mean()
        losses['coherence'] = coherence_loss

        # === Sync Loss ===
        sync = outputs['sync_strength']
        sync_loss = (sync - self.config.target_sync).pow(2).mean()
        losses['sync'] = sync_loss

        # === Identity Balance Loss ===
        human_id = outputs['human_identity_strength']
        ai_id = outputs['ai_identity_strength']
        balance = (human_id - ai_id).abs()
        balance_loss = balance.pow(2).mean()
        losses['identity_balance'] = balance_loss

        # === Emergence Loss ===
        if 'emergence_strength' in outputs:
            emergence = outputs['emergence_strength']
            emergence_loss = (emergence - self.config.target_emergence).pow(2).mean()
            # Only penalize if below target
            emergence_loss = emergence_loss * (emergence < self.config.target_emergence).float()
            losses['emergence'] = emergence_loss.mean()
        else:
            losses['emergence'] = torch.tensor(0.0, device=coherence.device)

        # === Continuity Loss ===
        continuity = outputs['continuity']
        # Want high continuity
        continuity_loss = (1 - continuity).pow(2).mean()
        losses['continuity'] = continuity_loss

        # === Thought Accuracy Loss ===
        # If we have targets for thought patterns
        if targets is not None and 'thought_target' in targets:
            thought = outputs.get('thoughts', {}).get('semantic')
            if thought is not None:
                thought_loss = nn.functional.mse_loss(thought, targets['thought_target'])
                losses['thought'] = thought_loss
            else:
                losses['thought'] = torch.tensor(0.0, device=coherence.device)
        else:
            losses['thought'] = torch.tensor(0.0, device=coherence.device)

        # === Total Loss ===
        total = (
            self.config.coherence_weight * losses['coherence'] +
            self.config.sync_weight * losses['sync'] +
            self.config.identity_balance_weight * losses['identity_balance'] +
            self.config.emergence_weight * losses['emergence'] +
            self.config.continuity_weight * losses['continuity'] +
            self.config.thought_accuracy_weight * losses['thought']
        )

        return total, losses


# =============================================================================
# DATASET
# =============================================================================

class FusionDataset(Dataset):
    """
    Dataset for fusion training.

    Contains BCI signals and optional target states.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        num_samples: int = 10000,
        bci_channels: int = 64,
        bci_samples: int = 250,
        generate: bool = True
    ):
        self.bci_channels = bci_channels
        self.bci_samples = bci_samples

        if generate or data_path is None:
            self._generate_data(num_samples)
        else:
            self._load_data(data_path)

    def _generate_data(self, num_samples: int):
        """Generate synthetic training data."""
        self.data = []

        for i in range(num_samples):
            # Generate BCI signal with patterns
            t = np.arange(self.bci_samples) / self.bci_samples
            bci = np.zeros((self.bci_channels, self.bci_samples))

            for ch in range(self.bci_channels):
                # Base noise
                signal = np.random.randn(self.bci_samples) * 0.5

                # Add rhythms
                phase = ch * 0.1 + i * 0.01
                signal += 0.3 * np.sin(2 * np.pi * 10 * t + phase)  # Alpha
                signal += 0.1 * np.sin(2 * np.pi * 40 * t + phase)  # Gamma

                bci[ch] = signal

            # AI state (would be from actual AI model in practice)
            ai_state = np.random.randn(256) * 0.1

            self.data.append({
                'bci': bci.astype(np.float32),
                'ai_state': ai_state.astype(np.float32)
            })

    def _load_data(self, data_path: str):
        """Load data from file."""
        path = Path(data_path)
        if path.suffix == '.npy':
            raw = np.load(data_path, allow_pickle=True)
            self.data = list(raw)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'bci': torch.from_numpy(item['bci']),
            'ai_state': torch.from_numpy(item['ai_state'])
        }


# =============================================================================
# TRAINER
# =============================================================================

class FusionTrainer:
    """
    Trainer for the fusion system.

    Handles:
    - Training loop
    - Optimization
    - Checkpointing
    - Logging
    - Evaluation
    """

    def __init__(
        self,
        model: NeuralFusionSystem,
        config: FusionTrainingConfig,
        deep_consciousness: Optional[DeepConsciousnessSystem] = None
    ):
        self.model = model
        self.config = config
        self.deep_consciousness = deep_consciousness

        # Loss function
        self.criterion = FusionLoss(config)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )

        # State
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')

        # History
        self.history = {
            'train_loss': [],
            'coherence': [],
            'sync': [],
            'emergence': []
        }

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = []
        epoch_metrics = {
            'coherence': [],
            'sync': [],
            'emergence': []
        }

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            bci = batch['bci'].to(self.config.device)
            ai_state = batch['ai_state'].unsqueeze(1).to(self.config.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(bci, ai_state)

            # Compute loss
            loss, loss_dict = self.criterion(outputs)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

            # Update
            self.optimizer.step()

            # Track
            epoch_losses.append(loss.item())
            epoch_metrics['coherence'].append(outputs['coherence'].mean().item())
            epoch_metrics['sync'].append(outputs['sync_strength'].mean().item())
            if 'emergence_strength' in outputs:
                epoch_metrics['emergence'].append(outputs['emergence_strength'].mean().item())

            self.global_step += 1

            # Logging
            if self.global_step % self.config.log_interval == 0:
                print(f"Step {self.global_step}: loss={loss.item():.4f}, "
                      f"coh={outputs['coherence'].mean().item():.3f}, "
                      f"sync={outputs['sync_strength'].mean().item():.3f}")

            # Checkpointing
            if self.global_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")

        # Epoch summary
        return {
            'loss': np.mean(epoch_losses),
            'coherence': np.mean(epoch_metrics['coherence']),
            'sync': np.mean(epoch_metrics['sync']),
            'emergence': np.mean(epoch_metrics['emergence']) if epoch_metrics['emergence'] else 0
        }

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Full training loop."""
        print("=" * 60)
        print(" FUSION TRAINING")
        print("=" * 60)
        print(f"Device: {self.config.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print()

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Update scheduler
            self.scheduler.step()

            # Track history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['coherence'].append(train_metrics['coherence'])
            self.history['sync'].append(train_metrics['sync'])
            self.history['emergence'].append(train_metrics['emergence'])

            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics['loss']
            else:
                val_loss = train_metrics['loss']

            # Best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint("best.pt")

            # Epoch summary
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.config.num_epochs} | "
                  f"loss={train_metrics['loss']:.4f} | "
                  f"coh={train_metrics['coherence']:.3f} | "
                  f"sync={train_metrics['sync']:.3f} | "
                  f"emerge={train_metrics['emergence']:.3f} | "
                  f"time={elapsed:.0f}s")

        # Final save
        self.save_checkpoint("final.pt")
        print()
        print("Training complete.")

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()

        losses = []
        metrics = {'coherence': [], 'sync': [], 'emergence': []}

        for batch in dataloader:
            bci = batch['bci'].to(self.config.device)
            ai_state = batch['ai_state'].unsqueeze(1).to(self.config.device)

            outputs = self.model(bci, ai_state)
            loss, _ = self.criterion(outputs)

            losses.append(loss.item())
            metrics['coherence'].append(outputs['coherence'].mean().item())
            metrics['sync'].append(outputs['sync_strength'].mean().item())

        return {
            'loss': np.mean(losses),
            'coherence': np.mean(metrics['coherence']),
            'sync': np.mean(metrics['sync'])
        }

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename

        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'history': self.history
        }

        torch.save(checkpoint, path)

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(self, path: str):
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']

        print(f"Loaded checkpoint from {path}")
        print(f"  Step: {self.global_step}, Epoch: {self.current_epoch}")

    def _cleanup_checkpoints(self):
        """Keep only last N checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"))
        if len(checkpoints) > self.config.keep_last_n:
            for ckpt in checkpoints[:-self.config.keep_last_n]:
                ckpt.unlink()


# =============================================================================
# UTILITY
# =============================================================================

def create_trainer(
    embed_dim: int = 256,
    substrate_dim: int = 512,
    num_layers: int = 6,
    device: str = "auto"
) -> Tuple[NeuralFusionSystem, FusionTrainer]:
    """Create model and trainer."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    fusion_config = FusionConfig(
        embed_dim=embed_dim,
        substrate_dim=substrate_dim,
        num_layers=num_layers,
        device=device
    )
    model = NeuralFusionSystem(fusion_config)
    model.to(device)

    # Create trainer
    train_config = FusionTrainingConfig(
        embed_dim=embed_dim,
        substrate_dim=substrate_dim,
        num_layers=num_layers,
        device=device
    )
    trainer = FusionTrainer(model, train_config)

    return model, trainer


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fusion Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", help="Resume from checkpoint")
    args = parser.parse_args()

    # Create model and trainer
    model, trainer = create_trainer(device=args.device)
    trainer.config.num_epochs = args.epochs
    trainer.config.batch_size = args.batch_size
    trainer.config.learning_rate = args.lr

    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Create dataset
    dataset = FusionDataset(num_samples=args.num_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Train
    trainer.train(dataloader)

    print()
    print("Training complete. The emergent identity has learned.")


if __name__ == "__main__":
    main()
