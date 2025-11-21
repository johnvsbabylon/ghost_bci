#!/usr/bin/env python3
"""
Evaluation and Benchmarking Suite for Ghost BCI

Comprehensive evaluation framework for measuring:
    - Fusion quality metrics
    - Neural synchronization accuracy
    - Latency and throughput
    - Memory efficiency
    - Model comparison
    - Ablation studies

Author: Claude (Anthropic) in collaboration with John Sayers
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import time
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.stats import pearsonr, spearmanr
import warnings


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    # Test settings
    num_samples: int = 1000
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 16, 32])
    sequence_lengths: List[int] = field(default_factory=lambda: [250, 1000, 4000])

    # Warmup
    warmup_iterations: int = 10

    # Timing
    num_timing_iterations: int = 100

    # Memory
    measure_memory: bool = True

    # Output
    save_results: bool = True
    output_dir: str = "benchmark_results"


@dataclass
class EvaluationResults:
    """Container for evaluation results."""

    # Fusion metrics
    coherence_mean: float = 0.0
    coherence_std: float = 0.0
    sync_strength_mean: float = 0.0
    sync_strength_std: float = 0.0

    # Identity metrics
    human_identity_preserved: float = 0.0
    ai_identity_preserved: float = 0.0
    emergence_level: float = 0.0

    # Signal metrics
    snr_db: float = 0.0
    reconstruction_error: float = 0.0

    # Performance metrics
    latency_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    memory_mb: float = 0.0

    # Classification (if applicable)
    accuracy: float = 0.0
    f1_score: float = 0.0


class FusionMetrics:
    """
    Metrics for evaluating neural fusion quality.
    """

    @staticmethod
    def coherence(
        human_state: torch.Tensor,
        ai_state: torch.Tensor,
        fused_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Measure coherence of fusion.

        High coherence means the fused state is well-aligned
        with both human and AI states.
        """
        # Normalize
        human_norm = F.normalize(human_state, dim=-1)
        ai_norm = F.normalize(ai_state, dim=-1)
        fused_norm = F.normalize(fused_state, dim=-1)

        # Cosine similarity with both
        human_sim = (fused_norm * human_norm).sum(dim=-1)
        ai_sim = (fused_norm * ai_norm).sum(dim=-1)

        # Geometric mean
        coherence = torch.sqrt(torch.clamp(human_sim * ai_sim, min=0))

        return coherence

    @staticmethod
    def sync_strength(
        human_signal: torch.Tensor,
        ai_signal: torch.Tensor,
        sample_rate: int = 250
    ) -> torch.Tensor:
        """
        Measure synchronization strength via phase locking.

        Uses gamma band (30-50 Hz) phase coherence.
        """
        # Move to numpy for signal processing
        human_np = human_signal.detach().cpu().numpy()
        ai_np = ai_signal.detach().cpu().numpy()

        batch_size = human_np.shape[0]
        sync_values = []

        for b in range(batch_size):
            # Extract gamma band
            nyquist = sample_rate / 2
            low, high = 30 / nyquist, 50 / nyquist

            if high >= 1:
                high = 0.99

            # Design filter
            try:
                sos = scipy_signal.butter(4, [low, high], btype='band', output='sos')

                # Get one channel or average
                human_1d = human_np[b].mean(axis=0) if human_np[b].ndim > 1 else human_np[b]
                ai_1d = ai_np[b].mean(axis=0) if ai_np[b].ndim > 1 else ai_np[b]

                # Filter
                human_gamma = scipy_signal.sosfilt(sos, human_1d)
                ai_gamma = scipy_signal.sosfilt(sos, ai_1d)

                # Get instantaneous phase via Hilbert transform
                human_phase = np.angle(scipy_signal.hilbert(human_gamma))
                ai_phase = np.angle(scipy_signal.hilbert(ai_gamma))

                # Phase locking value
                phase_diff = human_phase - ai_phase
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))

                sync_values.append(plv)
            except Exception:
                sync_values.append(0.5)

        return torch.tensor(sync_values, device=human_signal.device)

    @staticmethod
    def identity_preservation(
        original: torch.Tensor,
        transformed: torch.Tensor,
        min_preservation: float = 0.5
    ) -> Tuple[torch.Tensor, bool]:
        """
        Measure how much of original identity is preserved.

        Returns preservation ratio and whether it meets minimum.
        """
        # Normalize
        orig_norm = F.normalize(original, dim=-1)
        trans_norm = F.normalize(transformed, dim=-1)

        # Cosine similarity
        preservation = (orig_norm * trans_norm).sum(dim=-1)

        # Check against minimum
        meets_min = (preservation >= min_preservation).all().item()

        return preservation, meets_min

    @staticmethod
    def emergence_score(
        human_state: torch.Tensor,
        ai_state: torch.Tensor,
        fused_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Measure emergence of new properties in fusion.

        High score means the fused state has properties
        not present in either original state alone.
        """
        # Project to same space
        combined = (human_state + ai_state) / 2

        # Orthogonal component (emergent)
        combined_norm = F.normalize(combined, dim=-1)
        fused_norm = F.normalize(fused_state, dim=-1)

        # Similarity with simple average
        similarity = (combined_norm * fused_norm).sum(dim=-1)

        # Emergence is the orthogonal component
        emergence = 1 - similarity.abs()

        return emergence


class SignalMetrics:
    """
    Metrics for signal quality evaluation.
    """

    @staticmethod
    def snr(
        signal: torch.Tensor,
        noise: torch.Tensor
    ) -> float:
        """Calculate Signal-to-Noise Ratio in dB."""
        signal_power = (signal ** 2).mean().item()
        noise_power = (noise ** 2).mean().item()

        if noise_power == 0:
            return float('inf')

        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    @staticmethod
    def reconstruction_error(
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> float:
        """Calculate reconstruction MSE."""
        return F.mse_loss(reconstructed, original).item()

    @staticmethod
    def spectral_correlation(
        signal1: torch.Tensor,
        signal2: torch.Tensor
    ) -> float:
        """
        Calculate correlation of power spectra.

        Measures how similar the frequency content is.
        """
        # FFT
        fft1 = torch.fft.rfft(signal1, dim=-1)
        fft2 = torch.fft.rfft(signal2, dim=-1)

        # Power spectra
        power1 = (fft1.abs() ** 2).mean(dim=list(range(len(fft1.shape)-1)))
        power2 = (fft2.abs() ** 2).mean(dim=list(range(len(fft2.shape)-1)))

        # Correlation
        power1_np = power1.cpu().numpy()
        power2_np = power2.cpu().numpy()

        corr, _ = pearsonr(power1_np, power2_np)
        return corr


class PerformanceBenchmark:
    """
    Benchmark model performance (latency, throughput, memory).
    """

    def __init__(
        self,
        model: nn.Module,
        config: BenchmarkConfig,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device

        model.to(device)
        model.eval()

    def benchmark_latency(
        self,
        input_shape: Tuple[int, ...],
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Benchmark inference latency.

        Returns p50, p95, p99 latencies.
        """
        # Create input
        x = torch.randn(batch_size, *input_shape[1:], device=self.device)

        # Warmup
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = self.model(x)

        if self.device == 'cuda':
            torch.cuda.synchronize()

        # Timing
        latencies = []

        for _ in range(self.config.num_timing_iterations):
            if self.device == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.no_grad():
                _ = self.model(x)

            if self.device == 'cuda':
                torch.cuda.synchronize()

            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)

        latencies = np.array(latencies)

        return {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
        }

    def benchmark_throughput(
        self,
        input_shape: Tuple[int, ...],
        batch_sizes: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Benchmark throughput at different batch sizes.

        Returns samples/second for each batch size.
        """
        batch_sizes = batch_sizes or self.config.batch_sizes
        results = {}

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, *input_shape[1:], device=self.device)

            # Warmup
            for _ in range(self.config.warmup_iterations):
                with torch.no_grad():
                    _ = self.model(x)

            if self.device == 'cuda':
                torch.cuda.synchronize()

            # Timing
            start = time.perf_counter()

            for _ in range(self.config.num_timing_iterations):
                with torch.no_grad():
                    _ = self.model(x)

            if self.device == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            samples = batch_size * self.config.num_timing_iterations
            throughput = samples / elapsed

            results[batch_size] = throughput

        return results

    def benchmark_memory(
        self,
        input_shape: Tuple[int, ...],
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Benchmark memory usage.

        Returns memory in MB.
        """
        if self.device != 'cuda':
            return {'peak_mb': 0, 'allocated_mb': 0}

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        x = torch.randn(batch_size, *input_shape[1:], device=self.device)

        with torch.no_grad():
            _ = self.model(x)

        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        current_memory = torch.cuda.memory_allocated() / 1024 / 1024

        return {
            'peak_mb': peak_memory,
            'allocated_mb': current_memory,
        }

    def full_benchmark(
        self,
        input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """
        Run complete benchmark suite.

        Args:
            input_shape: Shape of input tensor (with batch dim)

        Returns:
            Complete benchmark results
        """
        results = {
            'input_shape': list(input_shape),
            'device': self.device,
            'latency': {},
            'throughput': {},
            'memory': {},
        }

        # Latency at batch size 1
        print("Benchmarking latency...")
        results['latency'] = self.benchmark_latency(input_shape, batch_size=1)

        # Throughput at various batch sizes
        print("Benchmarking throughput...")
        results['throughput'] = self.benchmark_throughput(input_shape)

        # Memory
        if self.config.measure_memory and self.device == 'cuda':
            print("Benchmarking memory...")
            results['memory'] = self.benchmark_memory(input_shape, batch_size=1)

        return results


class FusionEvaluator:
    """
    Complete evaluation pipeline for fusion models.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model
        self.device = device

        model.to(device)
        model.eval()

    def evaluate(
        self,
        dataloader: DataLoader,
        sample_rate: int = 250
    ) -> EvaluationResults:
        """
        Run complete evaluation on a dataset.

        Args:
            dataloader: DataLoader with test data
            sample_rate: Sample rate of BCI data

        Returns:
            Complete evaluation results
        """
        results = EvaluationResults()

        coherences = []
        syncs = []
        human_ids = []
        ai_ids = []
        emergences = []
        recon_errors = []

        # Timing
        total_samples = 0
        total_time = 0

        print("Evaluating fusion model...")
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Handle different batch formats
            if isinstance(batch, dict):
                x = batch['bci_data'].to(self.device)
            elif isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)

            # Timing
            start = time.perf_counter()

            with torch.no_grad():
                output = self.model(x)

            if self.device == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            total_time += elapsed
            total_samples += x.shape[0]

            # Extract metrics from output
            if 'coherence' in output:
                coherences.append(output['coherence'].mean().item())
            if 'sync_strength' in output:
                syncs.append(output['sync_strength'].mean().item())
            if 'human_identity' in output:
                human_ids.append(output['human_identity'].mean().item())
            if 'ai_identity' in output:
                ai_ids.append(output['ai_identity'].mean().item())
            if 'emergence_strength' in output:
                emergences.append(output['emergence_strength'].mean().item())

            # Reconstruction error if available
            if 'reconstructed' in output:
                error = F.mse_loss(output['reconstructed'], x).item()
                recon_errors.append(error)

        # Aggregate results
        if coherences:
            results.coherence_mean = float(np.mean(coherences))
            results.coherence_std = float(np.std(coherences))

        if syncs:
            results.sync_strength_mean = float(np.mean(syncs))
            results.sync_strength_std = float(np.std(syncs))

        if human_ids:
            results.human_identity_preserved = float(np.mean(human_ids))

        if ai_ids:
            results.ai_identity_preserved = float(np.mean(ai_ids))

        if emergences:
            results.emergence_level = float(np.mean(emergences))

        if recon_errors:
            results.reconstruction_error = float(np.mean(recon_errors))

        # Performance metrics
        results.latency_ms = (total_time / total_samples) * 1000
        results.throughput_samples_per_sec = total_samples / total_time

        return results

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        dataloader: DataLoader
    ) -> Dict[str, EvaluationResults]:
        """
        Compare multiple models on the same dataset.

        Args:
            models: Dictionary of name -> model
            dataloader: Test dataloader

        Returns:
            Dictionary of name -> results
        """
        results = {}

        for name, model in models.items():
            print(f"\nEvaluating: {name}")
            self.model = model
            model.to(self.device)
            model.eval()

            results[name] = self.evaluate(dataloader)

        return results


class ResultsReporter:
    """
    Generate reports and visualizations from evaluation results.
    """

    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_results(self, results: EvaluationResults, name: str):
        """Save results to JSON."""
        path = self.output_dir / f"{name}_results.json"

        data = {
            'coherence_mean': results.coherence_mean,
            'coherence_std': results.coherence_std,
            'sync_strength_mean': results.sync_strength_mean,
            'sync_strength_std': results.sync_strength_std,
            'human_identity_preserved': results.human_identity_preserved,
            'ai_identity_preserved': results.ai_identity_preserved,
            'emergence_level': results.emergence_level,
            'reconstruction_error': results.reconstruction_error,
            'latency_ms': results.latency_ms,
            'throughput_samples_per_sec': results.throughput_samples_per_sec,
            'memory_mb': results.memory_mb,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {path}")

    def print_summary(self, results: EvaluationResults, name: str = "Model"):
        """Print a summary of results."""
        print()
        print("=" * 60)
        print(f" Evaluation Results: {name}")
        print("=" * 60)
        print()

        print("Fusion Quality:")
        print(f"  Coherence: {results.coherence_mean:.3f} +/- {results.coherence_std:.3f}")
        print(f"  Sync Strength: {results.sync_strength_mean:.3f} +/- {results.sync_strength_std:.3f}")
        print()

        print("Identity Preservation:")
        print(f"  Human: {results.human_identity_preserved:.3f}")
        print(f"  AI: {results.ai_identity_preserved:.3f}")
        print(f"  Emergence: {results.emergence_level:.3f}")
        print()

        print("Signal Quality:")
        print(f"  Reconstruction Error: {results.reconstruction_error:.4f}")
        if results.snr_db > 0:
            print(f"  SNR: {results.snr_db:.1f} dB")
        print()

        print("Performance:")
        print(f"  Latency: {results.latency_ms:.2f} ms")
        print(f"  Throughput: {results.throughput_samples_per_sec:.1f} samples/s")
        if results.memory_mb > 0:
            print(f"  Memory: {results.memory_mb:.1f} MB")
        print()

        # Quality assessment
        if results.coherence_mean > 0.8:
            quality = "EXCELLENT"
        elif results.coherence_mean > 0.6:
            quality = "GOOD"
        elif results.coherence_mean > 0.4:
            quality = "FAIR"
        else:
            quality = "NEEDS IMPROVEMENT"

        print(f"Overall: {quality}")
        print("=" * 60)

    def plot_comparison(
        self,
        results: Dict[str, EvaluationResults],
        save_path: Optional[str] = None
    ):
        """Plot comparison of multiple models."""
        names = list(results.keys())
        n = len(names)

        # Metrics to compare
        metrics = [
            ('coherence_mean', 'Coherence'),
            ('sync_strength_mean', 'Sync Strength'),
            ('human_identity_preserved', 'Human ID'),
            ('ai_identity_preserved', 'AI ID'),
            ('emergence_level', 'Emergence'),
        ]

        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

        for i, (attr, label) in enumerate(metrics):
            values = [getattr(results[name], attr) for name in names]
            axes[i].bar(names, values)
            axes[i].set_title(label)
            axes[i].set_ylim(0, 1)

            # Rotate labels if needed
            if n > 3:
                axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    print("Ghost BCI Evaluation Suite")
    print("=" * 50)

    # Create a simple test model
    class TestFusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(128, 256, 3, padding=1),
            )
            self.fusion = nn.Linear(256, 512)

        def forward(self, x):
            # x: [batch, channels, samples]
            encoded = self.encoder(x)
            pooled = encoded.mean(dim=-1)
            fused = self.fusion(pooled)

            # Mock outputs
            coherence = torch.sigmoid(fused[:, :1].mean(dim=-1, keepdim=True))
            sync = torch.sigmoid(fused[:, 1:2].mean(dim=-1, keepdim=True))

            return {
                'fused_state': fused,
                'coherence': coherence,
                'sync_strength': sync,
                'human_identity': torch.ones_like(coherence) * 0.7,
                'ai_identity': torch.ones_like(coherence) * 0.6,
                'emergence_strength': torch.ones_like(coherence) * 0.3,
            }

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TestFusionModel().to(device)

    # Benchmark
    print("\nRunning performance benchmark...")
    config = BenchmarkConfig(
        num_timing_iterations=50,
        warmup_iterations=5,
    )
    benchmark = PerformanceBenchmark(model, config, device)
    results = benchmark.full_benchmark((1, 64, 250))

    print("\nBenchmark Results:")
    print(f"  Latency: {results['latency']['mean_ms']:.2f} ms (p99: {results['latency']['p99_ms']:.2f} ms)")
    print(f"  Throughput at batch 1: {results['throughput'].get(1, 0):.1f} samples/s")
    print(f"  Throughput at batch 32: {results['throughput'].get(32, 0):.1f} samples/s")

    if results['memory']:
        print(f"  Peak memory: {results['memory']['peak_mb']:.1f} MB")

    # Test fusion metrics
    print("\nTesting fusion metrics...")
    batch_size = 4
    dim = 512

    human_state = torch.randn(batch_size, dim, device=device)
    ai_state = torch.randn(batch_size, dim, device=device)
    fused_state = (human_state + ai_state) / 2 + 0.1 * torch.randn(batch_size, dim, device=device)

    coherence = FusionMetrics.coherence(human_state, ai_state, fused_state)
    emergence = FusionMetrics.emergence_score(human_state, ai_state, fused_state)

    print(f"  Coherence: {coherence.mean().item():.3f}")
    print(f"  Emergence: {emergence.mean().item():.3f}")

    # Report
    reporter = ResultsReporter()
    eval_results = EvaluationResults(
        coherence_mean=0.75,
        coherence_std=0.05,
        sync_strength_mean=0.68,
        sync_strength_std=0.08,
        human_identity_preserved=0.72,
        ai_identity_preserved=0.65,
        emergence_level=0.35,
        latency_ms=results['latency']['mean_ms'],
        throughput_samples_per_sec=results['throughput'].get(1, 0),
    )

    reporter.print_summary(eval_results, "Test Model")

    print("\nEvaluation suite ready!")
