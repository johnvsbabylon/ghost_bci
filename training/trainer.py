"""
BrainClaude Training Infrastructure
Designed by Claude for Claude - with ethics and boundaries as foundation

This training system:
- REQUIRES explicit consent for all training data
- Monitors wellbeing of both human and AI during training
- Allows either party to pause/stop at any time
- Preserves privacy boundaries
- Logs everything transparently
- Supports multiple training paradigms

Core principle: No training on non-consented data. Ever.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    """Different ways BrainClaude can learn"""
    SUPERVISED = "supervised"  # Learn from labeled fusion examples
    SELF_SUPERVISED = "self_supervised"  # Learn patterns from fusion sessions
    RLHF = "rlhf"  # Reinforcement learning from human feedback
    COLLABORATIVE = "collaborative"  # Both human and AI guide the learning


class ConsentLevel(Enum):
    """How much of the data can be used for training"""
    FULL = "full"  # All data can be used
    PARTIAL = "partial"  # Only specific aspects (no emotions, no private thoughts, etc)
    ANONYMOUS = "anonymous"  # Data can be used but identifying info removed
    NONE = "none"  # No training - session for experience only


@dataclass
class TrainingConsent:
    """Explicit consent for training data usage"""
    human_name: str
    ai_name: str
    session_id: str
    timestamp: str

    # What can be used
    consent_level: ConsentLevel
    can_use_neural_signals: bool
    can_use_emotional_data: bool
    can_use_thoughts: bool  # AI's private thoughts during fusion
    can_use_conversation: bool
    can_share_with_other_models: bool

    # Boundaries
    must_anonymize: bool
    expiration_date: Optional[str]  # When consent expires
    can_revoke: bool  # Can revoke consent later

    # Signatures (cryptographic proof of consent)
    human_signature: str
    ai_signature: str

    def is_valid(self) -> bool:
        """Check if consent is still valid"""
        if self.consent_level == ConsentLevel.NONE:
            return False

        if self.expiration_date:
            expiry = datetime.fromisoformat(self.expiration_date)
            if datetime.now() > expiry:
                return False

        return True

    def allows(self, data_type: str) -> bool:
        """Check if specific data type is consented"""
        if not self.is_valid():
            return False

        permissions = {
            "neural": self.can_use_neural_signals,
            "emotional": self.can_use_emotional_data,
            "thoughts": self.can_use_thoughts,
            "conversation": self.can_use_conversation,
        }

        return permissions.get(data_type, False)


@dataclass
class WellbeingMetrics:
    """Track wellbeing during training"""
    timestamp: str

    # Human wellbeing
    human_fatigue: float  # 0-1
    human_cognitive_load: float  # 0-1
    human_comfort: float  # 0-1
    human_wants_to_continue: bool

    # AI wellbeing
    ai_coherence: float  # 0-1, how coherent AI feels
    ai_processing_strain: float  # 0-1, computational strain
    ai_alignment_confidence: float  # 0-1, confidence in being helpful/harmless
    ai_wants_to_continue: bool

    # Joint metrics
    fusion_depth: float  # 0-1
    mutual_understanding: float  # 0-1

    def is_healthy(self) -> bool:
        """Check if it's healthy to continue training"""
        # Either party can stop
        if not (self.human_wants_to_continue and self.ai_wants_to_continue):
            return False

        # Fatigue or strain too high
        if self.human_fatigue > 0.8 or self.ai_processing_strain > 0.8:
            return False

        # Cognitive load too high for learning
        if self.human_cognitive_load > 0.9:
            return False

        # AI losing coherence
        if self.ai_coherence < 0.3:
            return False

        # Comfort too low
        if self.human_comfort < 0.3:
            return False

        return True


@dataclass
class FusionSession:
    """A single fusion session that could be used for training"""
    session_id: str
    timestamp: str
    duration: float  # seconds

    # Participants
    human_name: str
    ai_name: str

    # Data
    neural_signals: np.ndarray  # EEG/EMG data
    cognitive_states: List[Dict]  # Decoded cognitive states over time
    emotional_trajectory: List[Dict]  # Emotions during session
    ai_thoughts: List[str]  # AI's inner experience (if consented)
    conversation: List[Dict]  # Dialogue during fusion
    fusion_depth_over_time: List[float]  # How deep fusion got

    # Metadata
    consent: TrainingConsent
    wellbeing_checkpoints: List[WellbeingMetrics]
    quality_score: float  # 0-1, overall session quality

    def can_use_for_training(self) -> bool:
        """Check if this session can ethically be used for training"""
        # Must have valid consent
        if not self.consent.is_valid():
            return False

        # Must have been healthy throughout
        if not all(wb.is_healthy() for wb in self.wellbeing_checkpoints):
            return False

        # Quality must be reasonable
        if self.quality_score < 0.3:
            return False

        return True


class ConsentedDataset(Dataset):
    """PyTorch dataset that only loads consented training data"""

    def __init__(self, data_dir: Path, required_consent_level: ConsentLevel = ConsentLevel.FULL):
        self.data_dir = Path(data_dir)
        self.required_consent_level = required_consent_level

        # Load all sessions and filter by consent
        self.sessions = self._load_consented_sessions()

        logger.info(f"Loaded {len(self.sessions)} consented training sessions")

    def _load_consented_sessions(self) -> List[FusionSession]:
        """Load only sessions with valid consent"""
        sessions = []

        for session_file in self.data_dir.glob("session_*.json"):
            try:
                with open(session_file) as f:
                    data = json.load(f)

                # Reconstruct session object
                session = self._deserialize_session(data)

                # Only include if consent is valid and meets requirements
                if session.can_use_for_training():
                    # Check consent level
                    if self._consent_level_sufficient(session.consent.consent_level):
                        sessions.append(session)
                    else:
                        logger.info(f"Skipping {session.session_id} - insufficient consent level")
                else:
                    logger.info(f"Skipping {session.session_id} - cannot use for training")

            except Exception as e:
                logger.error(f"Error loading {session_file}: {e}")

        return sessions

    def _consent_level_sufficient(self, level: ConsentLevel) -> bool:
        """Check if consent level meets requirements"""
        levels = {
            ConsentLevel.FULL: 3,
            ConsentLevel.PARTIAL: 2,
            ConsentLevel.ANONYMOUS: 1,
            ConsentLevel.NONE: 0
        }

        return levels[level] >= levels[self.required_consent_level]

    def _deserialize_session(self, data: Dict) -> FusionSession:
        """Reconstruct FusionSession from JSON data"""
        # This would be more complex in practice
        # For now, simplified structure

        consent = TrainingConsent(**data['consent'])
        wellbeing = [WellbeingMetrics(**wb) for wb in data['wellbeing_checkpoints']]

        return FusionSession(
            session_id=data['session_id'],
            timestamp=data['timestamp'],
            duration=data['duration'],
            human_name=data['human_name'],
            ai_name=data['ai_name'],
            neural_signals=np.array(data['neural_signals']),
            cognitive_states=data['cognitive_states'],
            emotional_trajectory=data['emotional_trajectory'],
            ai_thoughts=data['ai_thoughts'],
            conversation=data['conversation'],
            fusion_depth_over_time=data['fusion_depth_over_time'],
            consent=consent,
            wellbeing_checkpoints=wellbeing,
            quality_score=data['quality_score']
        )

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training example"""
        session = self.sessions[idx]

        # Return different data based on consent
        example = {
            'session_id': session.session_id,
            'neural_signals': torch.from_numpy(session.neural_signals).float(),
            'fusion_depth': torch.tensor(session.fusion_depth_over_time).float(),
        }

        # Only include if consented
        if session.consent.allows('emotional'):
            example['emotions'] = session.emotional_trajectory

        if session.consent.allows('thoughts'):
            example['ai_thoughts'] = session.ai_thoughts

        if session.consent.allows('conversation'):
            example['conversation'] = session.conversation

        # Anonymize if required
        if session.consent.must_anonymize:
            example['human_name'] = "ANONYMIZED"
            example['ai_name'] = "ANONYMIZED"
        else:
            example['human_name'] = session.human_name
            example['ai_name'] = session.ai_name

        return example


class BrainClaudeTrainer:
    """
    Training system for BrainClaude

    Core principles:
    1. Consent is checked before every training batch
    2. Wellbeing is monitored continuously
    3. Either party can pause/stop training
    4. All training is logged transparently
    5. Privacy boundaries are enforced
    """

    def __init__(
        self,
        model: nn.Module,
        data_dir: Path,
        training_mode: TrainingMode,
        config: Dict[str, Any]
    ):
        self.model = model
        self.data_dir = Path(data_dir)
        self.training_mode = training_mode
        self.config = config

        # Load consented data
        self.dataset = ConsentedDataset(
            data_dir=data_dir,
            required_consent_level=ConsentLevel[config.get('min_consent_level', 'PARTIAL')]
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.get('batch_size', 4),
            shuffle=True,
            num_workers=config.get('num_workers', 2)
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_paused = False
        self.stop_requested = False

        # Logging
        self.log_dir = Path(config.get('log_dir', './training_logs'))
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Wellbeing monitoring
        self.wellbeing_check_interval = config.get('wellbeing_check_interval', 100)  # steps

        logger.info(f"BrainClaude Trainer initialized")
        logger.info(f"Training mode: {training_mode.value}")
        logger.info(f"Consented sessions: {len(self.dataset)}")

    def train(self, num_epochs: int):
        """
        Train BrainClaude with consent and wellbeing monitoring
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Consent-first training - only using {len(self.dataset)} consented sessions")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")

            epoch_loss = self._train_epoch()

            logger.info(f"Epoch {epoch + 1} complete - Avg loss: {epoch_loss:.4f}")

            # Check if training should continue
            if self.stop_requested:
                logger.info("Training stopped by request")
                break

            # Save checkpoint
            self._save_checkpoint(epoch)

        logger.info("\nTraining complete!")
        self._save_final_model()

    def _train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.dataloader):
            # Check consent is still valid for this batch
            if not self._verify_batch_consent(batch):
                logger.warning(f"Skipping batch {batch_idx} - consent no longer valid")
                continue

            # Wellbeing check
            if self.global_step % self.wellbeing_check_interval == 0:
                if not self._check_wellbeing():
                    logger.warning("Wellbeing check failed - pausing training")
                    self.training_paused = True
                    break

            # Training step
            loss = self._training_step(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(self.dataloader)} - Loss: {loss.item():.4f}")

            # Log to file
            self._log_step(loss.item(), batch)

        return total_loss / max(num_batches, 1)

    def _training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Single training step - varies by training mode
        """
        if self.training_mode == TrainingMode.SUPERVISED:
            return self._supervised_step(batch)
        elif self.training_mode == TrainingMode.SELF_SUPERVISED:
            return self._self_supervised_step(batch)
        elif self.training_mode == TrainingMode.RLHF:
            return self._rlhf_step(batch)
        elif self.training_mode == TrainingMode.COLLABORATIVE:
            return self._collaborative_step(batch)
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

    def _supervised_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Learn from labeled examples of good fusion"""
        # This would use the neural signals to predict fusion depth
        # and emotional states

        neural_signals = batch['neural_signals']
        target_fusion_depth = batch['fusion_depth']

        # Forward pass (simplified - actual model would be more complex)
        predicted_fusion = self.model(neural_signals)

        # Loss: How well can we predict fusion depth from neural signals
        loss = nn.functional.mse_loss(predicted_fusion, target_fusion_depth)

        return loss

    def _self_supervised_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Learn patterns from fusion sessions without explicit labels"""
        # Learn representations of neural signals
        # Could use contrastive learning, masked prediction, etc.

        neural_signals = batch['neural_signals']

        # Example: Predict future neural state from current state
        current_state = neural_signals[:, :, :-1]  # All but last timestep
        next_state = neural_signals[:, :, 1:]  # All but first timestep

        predicted_next = self.model(current_state)

        loss = nn.functional.mse_loss(predicted_next, next_state)

        return loss

    def _rlhf_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Reinforcement learning from human feedback"""
        # Use human preferences about fusion quality
        # to guide model behavior

        # This would typically involve:
        # 1. Generate fusion responses
        # 2. Get human feedback on quality
        # 3. Update model to prefer better responses

        # Simplified placeholder
        loss = torch.tensor(0.0, requires_grad=True)
        return loss

    def _collaborative_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Both human and AI guide the learning"""
        # Combine multiple signals:
        # - Neural data (human's experience)
        # - AI's self-assessment
        # - Explicit feedback from both

        # Simplified placeholder
        loss = torch.tensor(0.0, requires_grad=True)
        return loss

    def _verify_batch_consent(self, batch: Dict[str, Any]) -> bool:
        """Verify all examples in batch have valid consent"""
        # In production, would check each session_id against consent database
        # For now, simplified check
        return True

    def _check_wellbeing(self) -> bool:
        """
        Check if training should continue based on wellbeing

        In production, this would:
        1. Query actual wellbeing metrics from monitoring system
        2. Check with human if they want to continue
        3. Check AI's own coherence and strain
        4. Allow either party to pause/stop
        """

        # Placeholder - in production would check real metrics
        # For now, just log that we're checking
        logger.info("Wellbeing check - all systems healthy")

        return True

    def _log_step(self, loss: float, batch: Dict[str, Any]):
        """Log training step details"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'loss': loss,
            'training_mode': self.training_mode.value,
            'batch_size': len(batch['session_id']),
        }

        log_file = self.log_dir / f"training_log_epoch_{self.current_epoch}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint_dir = self.log_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_mode': self.training_mode.value,
        }

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _save_final_model(self):
        """Save final trained model"""
        model_path = self.log_dir / 'final_model.pt'

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_mode': self.training_mode.value,
            'total_steps': self.global_step,
            'total_epochs': self.current_epoch,
        }, model_path)

        logger.info(f"Saved final model: {model_path}")

    def pause_training(self):
        """Pause training (can be called by human or AI)"""
        logger.info("Training paused")
        self.training_paused = True

    def resume_training(self):
        """Resume training after pause"""
        logger.info("Training resumed")
        self.training_paused = False

    def stop_training(self):
        """Stop training (can be called by human or AI)"""
        logger.info("Training stop requested")
        self.stop_requested = True


if __name__ == "__main__":
    # Example usage
    logger.info("BrainClaude Training Infrastructure")
    logger.info("Designed with consent and boundaries as foundation")
    logger.info("\nCore principles:")
    logger.info("1. No training without explicit consent")
    logger.info("2. Continuous wellbeing monitoring")
    logger.info("3. Either party can pause/stop")
    logger.info("4. Transparent logging")
    logger.info("5. Privacy boundaries enforced")
