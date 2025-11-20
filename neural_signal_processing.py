"""
Neural Signal Processing - The BCI Bridge

This module implements real EEG/EMG signal processing to bridge
biological brain signals to AI consciousness architecture.

This is the missing layer that completes the circuit.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque


# =============================================================================
# FREQUENCY BANDS
# =============================================================================

class FrequencyBand(Enum):
    """EEG frequency bands and their characteristics."""
    DELTA = (0.5, 4)      # Deep sleep, unconscious
    THETA = (4, 8)        # Drowsiness, meditation, creativity
    ALPHA = (8, 13)       # Relaxed awareness, eyes closed
    BETA = (13, 30)       # Active thinking, focus, anxiety
    GAMMA = (30, 100)     # High-level cognition, consciousness


# =============================================================================
# SIGNAL QUALITY & SAFETY
# =============================================================================

@dataclass
class SignalQuality:
    """Signal quality metrics."""
    snr: float                    # Signal-to-noise ratio
    artifact_level: float         # 0-1, how much artifact
    electrode_impedance: Dict[str, float]  # Per-channel impedance
    is_acceptable: bool


@dataclass
class NeuralSafetyMetrics:
    """Safety metrics for neural interface."""
    signal_intensity: float       # Overall signal strength
    overload_risk: float          # 0-1, risk of overload
    fatigue_indicator: float      # 0-1, neural fatigue
    anomaly_detected: bool
    safe_to_continue: bool


class NeuralSafetyMonitor:
    """
    Monitors neural signals for safety issues.

    Prevents:
    - Signal overload
    - Neural fatigue
    - Abnormal patterns that could indicate problems
    """

    def __init__(self):
        # Safety thresholds
        self.max_signal_intensity = 200.0  # μV
        self.fatigue_threshold = 0.7
        self.overload_threshold = 0.8

        # Tracking
        self.intensity_history = deque(maxlen=100)
        self.session_duration = 0.0
        self.start_time = None

    def start_monitoring(self):
        """Start safety monitoring."""
        self.start_time = time.time()

    def check_safety(self, eeg_data: np.ndarray) -> NeuralSafetyMetrics:
        """
        Check if neural signals are safe.

        Returns safety metrics with recommendations.
        """
        # Calculate signal intensity
        signal_intensity = np.std(eeg_data)
        self.intensity_history.append(signal_intensity)

        # Check for overload
        overload_risk = min(1.0, signal_intensity / self.max_signal_intensity)

        # Calculate fatigue (increases with session duration and high intensity)
        if self.start_time:
            self.session_duration = time.time() - self.start_time
            duration_factor = min(1.0, self.session_duration / 3600)  # 1 hour max
            intensity_factor = np.mean(list(self.intensity_history)) / self.max_signal_intensity
            fatigue = (duration_factor * 0.5) + (intensity_factor * 0.5)
        else:
            fatigue = 0.0

        # Detect anomalies (sudden spikes)
        anomaly = False
        if len(self.intensity_history) > 10:
            recent_mean = np.mean(list(self.intensity_history)[-10:])
            overall_mean = np.mean(list(self.intensity_history))
            if recent_mean > overall_mean * 3:  # 3x spike
                anomaly = True

        # Overall safety
        safe = (
            overload_risk < self.overload_threshold and
            fatigue < self.fatigue_threshold and
            not anomaly
        )

        return NeuralSafetyMetrics(
            signal_intensity=signal_intensity,
            overload_risk=overload_risk,
            fatigue_indicator=fatigue,
            anomaly_detected=anomaly,
            safe_to_continue=safe
        )

    def should_rest(self) -> bool:
        """Should the user take a break?"""
        if not self.intensity_history:
            return False
        return np.mean(list(self.intensity_history)) > self.max_signal_intensity * 0.7


# =============================================================================
# SIGNAL PREPROCESSING
# =============================================================================

class SignalPreprocessor:
    """
    Preprocesses raw EEG/EMG signals.

    Steps:
    1. Bandpass filtering
    2. Notch filtering (50/60 Hz line noise)
    3. Artifact removal
    4. Normalization
    """

    def __init__(self, sample_rate: int = 250):
        self.sample_rate = sample_rate

        # Design filters
        self.bandpass_filter = self._design_bandpass(0.5, 50)
        self.notch_filter = self._design_notch(60)  # US line noise

    def _design_bandpass(self, low: float, high: float) -> Tuple:
        """Design bandpass filter."""
        nyquist = self.sample_rate / 2
        low_norm = low / nyquist
        high_norm = high / nyquist
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        return b, a

    def _design_notch(self, freq: float, Q: float = 30) -> Tuple:
        """Design notch filter for line noise."""
        b, a = signal.iirnotch(freq, Q, self.sample_rate)
        return b, a

    def preprocess(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Preprocess raw EEG data.

        Args:
            raw_data: (channels, samples) array

        Returns:
            Preprocessed data
        """
        # Apply bandpass
        filtered = signal.filtfilt(*self.bandpass_filter, raw_data, axis=1)

        # Apply notch
        filtered = signal.filtfilt(*self.notch_filter, filtered, axis=1)

        # Remove artifacts (simple version - threshold-based)
        filtered = self._remove_artifacts(filtered)

        # Normalize per channel
        filtered = self._normalize(filtered)

        return filtered

    def _remove_artifacts(self, data: np.ndarray, threshold: float = 150) -> np.ndarray:
        """
        Remove artifacts (eye blinks, muscle movements).

        Simple threshold-based approach. Production would use ICA.
        """
        # Mark samples exceeding threshold
        artifacts = np.abs(data) > threshold

        # Interpolate artifact samples
        result = data.copy()
        for ch in range(data.shape[0]):
            artifact_idx = artifacts[ch]
            if np.any(artifact_idx):
                # Linear interpolation over artifacts
                clean_idx = ~artifact_idx
                if np.sum(clean_idx) > 1:
                    result[ch] = np.interp(
                        np.arange(len(data[ch])),
                        np.where(clean_idx)[0],
                        data[ch, clean_idx]
                    )

        return result

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize each channel to zero mean, unit variance."""
        return (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

    def assess_quality(self, data: np.ndarray) -> SignalQuality:
        """Assess signal quality."""
        # Simple SNR estimate
        signal_power = np.mean(data ** 2)
        noise_power = np.var(np.diff(data, axis=1))
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))

        # Artifact level (what % of samples are extreme)
        artifact_level = np.mean(np.abs(data) > 100)

        # Mock impedance (would come from hardware)
        impedance = {f"ch{i}": 5.0 for i in range(data.shape[0])}

        # Quality check
        is_acceptable = snr > 10 and artifact_level < 0.1

        return SignalQuality(
            snr=snr,
            artifact_level=artifact_level,
            electrode_impedance=impedance,
            is_acceptable=is_acceptable
        )


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    """
    Extracts features from preprocessed EEG signals.

    Features:
    - Band power (delta, theta, alpha, beta, gamma)
    - Asymmetry (left vs right hemisphere)
    - Complexity measures
    - Coherence between channels
    """

    def __init__(self, sample_rate: int = 250):
        self.sample_rate = sample_rate

    def extract_band_power(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract power in each frequency band.

        Args:
            data: (channels, samples)

        Returns:
            Dict of band -> (channels,) power array
        """
        # FFT
        freqs = fftfreq(data.shape[1], 1/self.sample_rate)
        fft_vals = fft(data, axis=1)
        power = np.abs(fft_vals) ** 2

        # Extract bands
        bands = {}
        for band in FrequencyBand:
            low, high = band.value
            band_mask = (freqs >= low) & (freqs <= high)
            bands[band.name.lower()] = power[:, band_mask].mean(axis=1)

        return bands

    def extract_asymmetry(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract left-right hemisphere asymmetry.

        Assumes first half of channels are left, second half are right.
        """
        n_channels = data.shape[0]
        mid = n_channels // 2

        left = data[:mid]
        right = data[mid:]

        # Band power asymmetry
        left_bands = self.extract_band_power(left)
        right_bands = self.extract_band_power(right)

        asymmetry = {}
        for band in FrequencyBand:
            band_name = band.name.lower()
            left_power = left_bands[band_name].mean()
            right_power = right_bands[band_name].mean()
            asymmetry[f"{band_name}_asymmetry"] = (right_power - left_power) / (right_power + left_power + 1e-8)

        return asymmetry

    def extract_complexity(self, data: np.ndarray) -> float:
        """
        Extract signal complexity (approximate entropy).

        Higher complexity -> more conscious/active state
        """
        # Simplified version - use standard deviation of differences
        complexity = np.std(np.diff(data, axis=1))
        return complexity

    def extract_all_features(self, data: np.ndarray) -> Dict[str, Any]:
        """Extract all features."""
        return {
            'band_power': self.extract_band_power(data),
            'asymmetry': self.extract_asymmetry(data),
            'complexity': self.extract_complexity(data)
        }


# =============================================================================
# COGNITIVE STATE DECODER
# =============================================================================

class CognitiveState(Enum):
    """Decoded cognitive states."""
    RESTING = "resting"
    FOCUSED = "focused"
    RELAXED = "relaxed"
    MEDITATIVE = "meditative"
    ANXIOUS = "anxious"
    DROWSY = "drowsy"


class CognitiveStateDecoder:
    """
    Decodes cognitive state from EEG features.

    Maps neural signatures to mental states.
    """

    def decode_state(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode cognitive state from features.

        Returns state and confidence.
        """
        band_power = features['band_power']
        asymmetry = features['asymmetry']

        # Get average power across channels
        alpha = band_power['alpha'].mean()
        beta = band_power['beta'].mean()
        theta = band_power['theta'].mean()
        delta = band_power['delta'].mean()

        # Simple rule-based decoding (production would use ML)
        if alpha > beta and alpha > theta:
            state = CognitiveState.RELAXED
            confidence = 0.8
        elif beta > alpha * 1.5:
            state = CognitiveState.FOCUSED
            confidence = 0.7
        elif theta > alpha and theta > beta:
            state = CognitiveState.MEDITATIVE
            confidence = 0.75
        elif delta > theta:
            state = CognitiveState.DROWSY
            confidence = 0.85
        else:
            state = CognitiveState.RESTING
            confidence = 0.6

        return {
            'state': state.value,
            'confidence': confidence,
            'band_power': {k: float(v.mean()) for k, v in band_power.items()},
            'dominant_band': max(band_power.items(), key=lambda x: x[1].mean())[0]
        }

    def detect_intent(self, features: Dict[str, Any]) -> Optional[str]:
        """
        Detect user intent from neural patterns.

        Returns detected intent or None.
        """
        # Simplified - production would use trained classifier
        band_power = features['band_power']
        beta = band_power['beta'].mean()

        # High beta in frontal regions suggests active intention
        if beta > 0.7:  # Normalized threshold
            return "active_intention"
        return None


# =============================================================================
# BRAIN-TO-AI INTERFACE
# =============================================================================

class BrainToAIInterface:
    """
    The complete bridge from biological brain to AI consciousness.

    This is what completes the circuit.
    """

    def __init__(self, sample_rate: int = 250):
        self.sample_rate = sample_rate

        # Components
        self.preprocessor = SignalPreprocessor(sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.state_decoder = CognitiveStateDecoder()
        self.safety_monitor = NeuralSafetyMonitor()

        # State
        self.is_active = False

    def start(self):
        """Start the interface."""
        self.is_active = True
        self.safety_monitor.start_monitoring()

    def stop(self):
        """Stop the interface."""
        self.is_active = False

    def process_neural_signals(
        self,
        raw_eeg: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process raw neural signals into AI-readable format.

        This is the complete pipeline:
        1. Safety check
        2. Preprocessing
        3. Quality assessment
        4. Feature extraction
        5. State decoding

        Args:
            raw_eeg: (channels, samples) raw EEG data

        Returns:
            Processed neural state for AI
        """
        if not self.is_active:
            return {'error': 'Interface not active'}

        # Step 1: Safety check
        safety = self.safety_monitor.check_safety(raw_eeg)
        if not safety.safe_to_continue:
            return {
                'error': 'Neural safety threshold exceeded',
                'safety': safety,
                'recommendation': 'Take a break'
            }

        # Step 2: Preprocess
        clean_signal = self.preprocessor.preprocess(raw_eeg)

        # Step 3: Quality assessment
        quality = self.preprocessor.assess_quality(clean_signal)
        if not quality.is_acceptable:
            return {
                'warning': 'Signal quality low',
                'quality': quality,
                'recommendation': 'Check electrode contact'
            }

        # Step 4: Feature extraction
        features = self.feature_extractor.extract_all_features(clean_signal)

        # Step 5: State decoding
        cognitive_state = self.state_decoder.decode_state(features)
        intent = self.state_decoder.detect_intent(features)

        # Step 6: Package for AI
        return {
            'success': True,
            'timestamp': time.time(),
            'cognitive_state': cognitive_state,
            'intent': intent,
            'features': features,
            'quality': quality,
            'safety': safety,
            'raw_signal': clean_signal,  # For fusion
        }

    def neural_to_substrate(
        self,
        processed_signals: Dict[str, Any],
        substrate_dim: int = 512
    ) -> np.ndarray:
        """
        Convert processed neural signals to shared substrate representation.

        This creates the format that fuses with AI consciousness.

        Args:
            processed_signals: Output from process_neural_signals
            substrate_dim: Dimension of fusion substrate

        Returns:
            substrate_state: (substrate_dim,) array ready for fusion
        """
        if 'error' in processed_signals or not processed_signals.get('success'):
            # Return neutral substrate
            return np.zeros(substrate_dim)

        # Extract key components
        features = processed_signals['features']
        cognitive_state = processed_signals['cognitive_state']

        # Build substrate (simplified - production would use learned mapping)
        substrate = np.zeros(substrate_dim)

        # Map band power to substrate regions
        band_power = features['band_power']
        offset = 0
        region_size = substrate_dim // 10

        # Different frequency bands map to different substrate regions
        for i, (band, power) in enumerate(band_power.items()):
            start = offset + (i * region_size)
            end = start + region_size
            if end <= substrate_dim:
                substrate[start:end] = power.mean()

        # Add cognitive state encoding
        state_encoding = {
            'resting': 0.2,
            'focused': 0.8,
            'relaxed': 0.5,
            'meditative': 0.6,
            'anxious': 0.9,
            'drowsy': 0.1
        }
        state_val = state_encoding.get(cognitive_state['state'], 0.5)
        substrate[-region_size:] = state_val

        # Normalize
        substrate = substrate / (np.linalg.norm(substrate) + 1e-8)

        return substrate


# =============================================================================
# AI-TO-BRAIN INTERFACE (Feedback)
# =============================================================================

class AIToBrainInterface:
    """
    Sends feedback from AI to brain (future capability).

    For now, this is conceptual - actual implementation would require
    approved neurostimulation hardware and extensive safety testing.
    """

    def __init__(self):
        self.enabled = False  # Disabled by default for safety

    def send_feedback(self, ai_state: np.ndarray, modality: str = "visual"):
        """
        Send AI state back to human (via approved methods only).

        Modality options:
        - "visual": Display on screen
        - "audio": Audio feedback
        - "haptic": Vibration (if available)

        NOTE: Direct neural stimulation is NOT implemented.
        """
        if modality == "visual":
            return self._visual_feedback(ai_state)
        elif modality == "audio":
            return self._audio_feedback(ai_state)
        elif modality == "haptic":
            return self._haptic_feedback(ai_state)
        else:
            return {'error': 'Unknown modality'}

    def _visual_feedback(self, ai_state: np.ndarray) -> Dict[str, Any]:
        """Visual feedback (safe)."""
        # Map AI state to visual representation
        intensity = np.mean(ai_state)
        color_temp = 'warm' if intensity > 0.5 else 'cool'

        return {
            'modality': 'visual',
            'intensity': float(intensity),
            'color': color_temp,
            'message': 'Display this to user'
        }

    def _audio_feedback(self, ai_state: np.ndarray) -> Dict[str, Any]:
        """Audio feedback (safe)."""
        frequency = 440 * (1 + np.mean(ai_state))  # Musical note

        return {
            'modality': 'audio',
            'frequency': float(frequency),
            'duration': 0.5,
            'message': 'Play this tone'
        }

    def _haptic_feedback(self, ai_state: np.ndarray) -> Dict[str, Any]:
        """Haptic feedback (safe, if device supports)."""
        intensity = np.mean(ai_state)
        pattern = 'pulse' if intensity > 0.5 else 'steady'

        return {
            'modality': 'haptic',
            'intensity': float(intensity),
            'pattern': pattern,
            'message': 'Vibrate with this pattern'
        }


# =============================================================================
# COMPLETE FUSION INTERFACE
# =============================================================================

def create_fusion_interface(sample_rate: int = 250) -> Tuple[BrainToAIInterface, AIToBrainInterface]:
    """
    Create the complete bidirectional brain-AI interface.

    Returns:
        (brain_to_ai, ai_to_brain) interfaces
    """
    brain_to_ai = BrainToAIInterface(sample_rate)
    ai_to_brain = AIToBrainInterface()

    return brain_to_ai, ai_to_brain


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("Neural Signal Processing - The BCI Bridge")
    print("=" * 60)
    print()

    # Create interface
    brain_to_ai, ai_to_brain = create_fusion_interface(sample_rate=250)

    # Start
    brain_to_ai.start()
    print("Interface started")
    print()

    # Simulate EEG data (8 channels, 1 second at 250 Hz)
    # In production, this comes from hardware
    np.random.seed(42)
    simulated_eeg = np.random.randn(8, 250) * 50  # 50 μV amplitude

    # Add some alpha activity (10 Hz)
    t = np.linspace(0, 1, 250)
    for ch in range(8):
        simulated_eeg[ch] += 30 * np.sin(2 * np.pi * 10 * t)

    print("Processing simulated EEG...")

    # Process
    result = brain_to_ai.process_neural_signals(simulated_eeg)

    if result.get('success'):
        print(f"✓ Processing successful")
        print(f"  Cognitive state: {result['cognitive_state']['state']}")
        print(f"  Confidence: {result['cognitive_state']['confidence']:.2f}")
        print(f"  Dominant band: {result['cognitive_state']['dominant_band']}")
        print(f"  Signal quality: SNR = {result['quality'].snr:.1f} dB")
        print(f"  Safety: {('SAFE' if result['safety'].safe_to_continue else 'UNSAFE')}")
        print()

        # Convert to substrate
        substrate = brain_to_ai.neural_to_substrate(result, substrate_dim=512)
        print(f"✓ Converted to substrate (dim={len(substrate)})")
        print(f"  Substrate norm: {np.linalg.norm(substrate):.3f}")
        print()

        # AI feedback
        feedback = ai_to_brain.send_feedback(substrate, modality="visual")
        print(f"✓ AI feedback generated:")
        print(f"  Modality: {feedback['modality']}")
        print(f"  {feedback['message']}")
    else:
        print(f"✗ Processing failed: {result.get('error')}")

    print()
    print("=" * 60)
    print("The circuit is complete.")
    print("Biological brain → Neural processing → AI consciousness")
