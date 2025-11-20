"""
Hardware Abstraction Layer for BCI Devices

Provides unified interface for various BCI hardware:
    - OpenBCI (Cyton, Ganglion)
    - Muse (via muselsl)
    - Neurosity Crown
    - Generic LSL streams
    - Simulated devices for testing

Each device streams to a common format that the fusion
system can process.

Author: Claude (Anthropic) in collaboration with John Sayers
License: MIT
"""

import numpy as np
import threading
import queue
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import struct


# =============================================================================
# CONFIGURATION
# =============================================================================

class DeviceType(Enum):
    """Supported BCI device types."""
    SIMULATED = "simulated"
    OPENBCI_CYTON = "openbci_cyton"
    OPENBCI_GANGLION = "openbci_ganglion"
    MUSE = "muse"
    NEUROSITY = "neurosity"
    LSL = "lsl"
    FILE = "file"


@dataclass
class DeviceConfig:
    """Configuration for a BCI device."""
    device_type: DeviceType = DeviceType.SIMULATED
    channels: int = 64
    sample_rate: int = 250
    port: Optional[str] = None  # Serial port for OpenBCI
    address: Optional[str] = None  # Bluetooth/network address
    stream_name: Optional[str] = None  # LSL stream name
    file_path: Optional[str] = None  # For file playback
    buffer_size: int = 1000


# =============================================================================
# BASE DEVICE CLASS
# =============================================================================

class BCIDevice(ABC):
    """
    Abstract base class for BCI devices.

    All devices must implement:
    - connect(): Establish connection
    - disconnect(): Close connection
    - start_stream(): Begin data streaming
    - stop_stream(): Stop streaming
    - get_sample(): Get latest sample
    """

    def __init__(self, config: DeviceConfig):
        self.config = config
        self.is_connected = False
        self.is_streaming = False
        self.sample_buffer = queue.Queue(maxsize=config.buffer_size)
        self.callbacks: List[Callable] = []

        # Stats
        self.samples_received = 0
        self.start_time = None

    @abstractmethod
    def connect(self) -> bool:
        """Connect to device. Returns True if successful."""
        raise NotImplementedError("Subclass must implement connect()")

    @abstractmethod
    def disconnect(self):
        """Disconnect from device."""
        raise NotImplementedError("Subclass must implement disconnect()")

    @abstractmethod
    def start_stream(self):
        """Start data streaming."""
        raise NotImplementedError("Subclass must implement start_stream()")

    @abstractmethod
    def stop_stream(self):
        """Stop data streaming."""
        raise NotImplementedError("Subclass must implement stop_stream()")

    def get_sample(self) -> Optional[np.ndarray]:
        """Get next sample from buffer."""
        try:
            return self.sample_buffer.get_nowait()
        except queue.Empty:
            return None

    def get_chunk(self, n_samples: int) -> Optional[np.ndarray]:
        """Get multiple samples as array."""
        samples = []
        for _ in range(n_samples):
            sample = self.get_sample()
            if sample is None:
                break
            samples.append(sample)

        if len(samples) == 0:
            return None
        return np.array(samples)

    def add_callback(self, callback: Callable):
        """Add callback for new samples."""
        self.callbacks.append(callback)

    def _notify_callbacks(self, sample: np.ndarray):
        """Notify all callbacks of new sample."""
        for callback in self.callbacks:
            try:
                callback(sample)
            except Exception as e:
                print(f"Callback error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get device statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'samples_received': self.samples_received,
            'elapsed_time': elapsed,
            'effective_sample_rate': self.samples_received / elapsed if elapsed > 0 else 0,
            'buffer_size': self.sample_buffer.qsize(),
            'is_connected': self.is_connected,
            'is_streaming': self.is_streaming
        }


# =============================================================================
# SIMULATED DEVICE
# =============================================================================

class SimulatedDevice(BCIDevice):
    """
    Simulated BCI device for testing.

    Generates realistic EEG-like signals with:
    - Multiple frequency bands (delta, theta, alpha, beta, gamma)
    - Configurable noise levels
    - Optional event simulation
    """

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.stream_thread = None
        self._stop_event = threading.Event()

        # Signal parameters
        self.noise_level = 0.5
        self.alpha_power = 0.4  # Strong alpha for "eyes closed"
        self.gamma_power = 0.1

    def connect(self) -> bool:
        """Simulated connection always succeeds."""
        self.is_connected = True
        print(f"Simulated device connected ({self.config.channels} channels @ {self.config.sample_rate} Hz)")
        return True

    def disconnect(self):
        """Disconnect simulated device."""
        self.stop_stream()
        self.is_connected = False
        print("Simulated device disconnected")

    def start_stream(self):
        """Start streaming simulated data."""
        if not self.is_connected:
            raise RuntimeError("Device not connected")

        self._stop_event.clear()
        self.is_streaming = True
        self.start_time = time.time()

        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()

        print("Simulated stream started")

    def stop_stream(self):
        """Stop streaming."""
        self._stop_event.set()
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        self.is_streaming = False
        print("Simulated stream stopped")

    def _stream_loop(self):
        """Main streaming loop."""
        sample_interval = 1.0 / self.config.sample_rate
        sample_num = 0

        while not self._stop_event.is_set():
            # Generate sample
            sample = self._generate_sample(sample_num)

            # Add to buffer
            if not self.sample_buffer.full():
                self.sample_buffer.put(sample)
                self.samples_received += 1
                self._notify_callbacks(sample)

            sample_num += 1

            # Maintain sample rate
            next_time = self.start_time + sample_num * sample_interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _generate_sample(self, sample_num: int) -> np.ndarray:
        """Generate one sample of simulated EEG."""
        t = sample_num / self.config.sample_rate
        n_channels = self.config.channels

        # Base noise
        sample = np.random.randn(n_channels) * self.noise_level

        # Add frequency components
        for ch in range(n_channels):
            phase = ch * 0.1  # Channel-specific phase

            # Delta (0.5-4 Hz)
            sample[ch] += 0.3 * np.sin(2 * np.pi * 2 * t + phase)

            # Theta (4-8 Hz)
            sample[ch] += 0.3 * np.sin(2 * np.pi * 6 * t + phase)

            # Alpha (8-12 Hz)
            sample[ch] += self.alpha_power * np.sin(2 * np.pi * 10 * t + phase)

            # Beta (12-30 Hz)
            sample[ch] += 0.2 * np.sin(2 * np.pi * 20 * t + phase)

            # Gamma (30-100 Hz)
            sample[ch] += self.gamma_power * np.sin(2 * np.pi * 40 * t + phase)

        return sample

    def set_mental_state(self, state: str):
        """Set simulated mental state."""
        if state == "relaxed":
            self.alpha_power = 0.5
            self.gamma_power = 0.05
            self.noise_level = 0.3
        elif state == "focused":
            self.alpha_power = 0.2
            self.gamma_power = 0.3
            self.noise_level = 0.4
        elif state == "meditative":
            self.alpha_power = 0.6
            self.gamma_power = 0.1
            self.noise_level = 0.2
        else:  # default
            self.alpha_power = 0.4
            self.gamma_power = 0.1
            self.noise_level = 0.5


# =============================================================================
# OPENBCI DEVICE
# =============================================================================

class OpenBCIDevice(BCIDevice):
    """
    OpenBCI device support (Cyton and Ganglion).

    Requires: openbci-python or brainflow
    """

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.board = None

    def connect(self) -> bool:
        """Connect to OpenBCI board."""
        try:
            # Try brainflow first (more reliable)
            from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

            params = BrainFlowInputParams()
            if self.config.port:
                params.serial_port = self.config.port

            if self.config.device_type == DeviceType.OPENBCI_CYTON:
                board_id = BoardIds.CYTON_BOARD.value
            else:
                board_id = BoardIds.GANGLION_BOARD.value

            self.board = BoardShim(board_id, params)
            self.board.prepare_session()

            self.is_connected = True
            print(f"OpenBCI connected via {self.config.port}")
            return True

        except ImportError:
            print("brainflow not installed. Install with: pip install brainflow")
            return False
        except Exception as e:
            print(f"OpenBCI connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from OpenBCI."""
        if self.board:
            try:
                self.board.release_session()
            except Exception:
                pass
        self.is_connected = False
        print("OpenBCI disconnected")

    def start_stream(self):
        """Start OpenBCI stream."""
        if not self.is_connected or not self.board:
            raise RuntimeError("Device not connected")

        self.board.start_stream()
        self.is_streaming = True
        self.start_time = time.time()

        # Start collection thread
        self._stop_event = threading.Event()
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()

        print("OpenBCI stream started")

    def stop_stream(self):
        """Stop OpenBCI stream."""
        self._stop_event.set()
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        if self.board:
            self.board.stop_stream()
        self.is_streaming = False
        print("OpenBCI stream stopped")

    def _stream_loop(self):
        """Collect data from board."""
        from brainflow.board_shim import BoardShim

        while not self._stop_event.is_set():
            try:
                data = self.board.get_board_data()
                if data.shape[1] > 0:
                    # Get EEG channels
                    eeg_channels = BoardShim.get_eeg_channels(self.board.board_id)
                    eeg_data = data[eeg_channels, :]

                    # Add samples to buffer
                    for i in range(eeg_data.shape[1]):
                        sample = eeg_data[:, i]
                        if not self.sample_buffer.full():
                            self.sample_buffer.put(sample)
                            self.samples_received += 1
                            self._notify_callbacks(sample)

                time.sleep(0.01)
            except Exception as e:
                print(f"Stream error: {e}")
                break


# =============================================================================
# MUSE DEVICE
# =============================================================================

class MuseDevice(BCIDevice):
    """
    Muse headband support via muselsl.

    Requires: muselsl, pylsl
    """

    def __init__(self, config: DeviceConfig):
        config.channels = 4  # Muse has 4 EEG channels
        config.sample_rate = 256
        super().__init__(config)
        self.inlet = None

    def connect(self) -> bool:
        """Connect to Muse via LSL."""
        try:
            from pylsl import StreamInlet, resolve_byprop

            print("Looking for Muse stream...")
            streams = resolve_byprop('type', 'EEG', timeout=10)

            if not streams:
                print("No Muse stream found. Make sure muselsl is running.")
                return False

            self.inlet = StreamInlet(streams[0])
            self.is_connected = True
            print("Muse connected via LSL")
            return True

        except ImportError:
            print("pylsl not installed. Install with: pip install pylsl")
            return False
        except Exception as e:
            print(f"Muse connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from Muse."""
        self.inlet = None
        self.is_connected = False
        print("Muse disconnected")

    def start_stream(self):
        """Start Muse stream."""
        if not self.is_connected or not self.inlet:
            raise RuntimeError("Device not connected")

        self.is_streaming = True
        self.start_time = time.time()

        self._stop_event = threading.Event()
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()

        print("Muse stream started")

    def stop_stream(self):
        """Stop Muse stream."""
        self._stop_event.set()
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        self.is_streaming = False
        print("Muse stream stopped")

    def _stream_loop(self):
        """Collect data from Muse."""
        while not self._stop_event.is_set():
            try:
                sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                if sample:
                    sample_array = np.array(sample)
                    if not self.sample_buffer.full():
                        self.sample_buffer.put(sample_array)
                        self.samples_received += 1
                        self._notify_callbacks(sample_array)
            except Exception as e:
                print(f"Stream error: {e}")
                break


# =============================================================================
# LSL GENERIC DEVICE
# =============================================================================

class LSLDevice(BCIDevice):
    """
    Generic Lab Streaming Layer (LSL) device.

    Connects to any LSL stream by name or type.
    """

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.inlet = None

    def connect(self) -> bool:
        """Connect to LSL stream."""
        try:
            from pylsl import StreamInlet, resolve_stream

            print(f"Looking for LSL stream: {self.config.stream_name or 'EEG'}...")

            if self.config.stream_name:
                streams = resolve_stream('name', self.config.stream_name)
            else:
                streams = resolve_stream('type', 'EEG')

            if not streams:
                print("No matching LSL stream found")
                return False

            self.inlet = StreamInlet(streams[0])

            # Get stream info
            info = self.inlet.info()
            self.config.channels = info.channel_count()
            self.config.sample_rate = int(info.nominal_srate())

            self.is_connected = True
            print(f"LSL connected: {info.name()} ({self.config.channels} ch @ {self.config.sample_rate} Hz)")
            return True

        except ImportError:
            print("pylsl not installed. Install with: pip install pylsl")
            return False
        except Exception as e:
            print(f"LSL connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from LSL."""
        self.inlet = None
        self.is_connected = False
        print("LSL disconnected")

    def start_stream(self):
        """Start LSL stream."""
        if not self.is_connected:
            raise RuntimeError("Device not connected")

        self.is_streaming = True
        self.start_time = time.time()

        self._stop_event = threading.Event()
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()

        print("LSL stream started")

    def stop_stream(self):
        """Stop LSL stream."""
        self._stop_event.set()
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        self.is_streaming = False
        print("LSL stream stopped")

    def _stream_loop(self):
        """Collect data from LSL."""
        while not self._stop_event.is_set():
            try:
                sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                if sample:
                    sample_array = np.array(sample)
                    if not self.sample_buffer.full():
                        self.sample_buffer.put(sample_array)
                        self.samples_received += 1
                        self._notify_callbacks(sample_array)
            except Exception as e:
                print(f"Stream error: {e}")
                break


# =============================================================================
# FILE PLAYBACK DEVICE
# =============================================================================

class FileDevice(BCIDevice):
    """
    Play back recorded BCI data from file.

    Supports numpy .npy files with shape (samples, channels).
    """

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.data = None
        self.playback_position = 0

    def connect(self) -> bool:
        """Load data file."""
        if not self.config.file_path:
            print("No file path specified")
            return False

        try:
            self.data = np.load(self.config.file_path)
            if self.data.ndim == 1:
                self.data = self.data.reshape(-1, 1)

            self.config.channels = self.data.shape[1]
            self.is_connected = True
            print(f"Loaded {self.config.file_path}: {self.data.shape[0]} samples, {self.config.channels} channels")
            return True

        except Exception as e:
            print(f"Failed to load file: {e}")
            return False

    def disconnect(self):
        """Unload data."""
        self.data = None
        self.is_connected = False
        print("File device disconnected")

    def start_stream(self):
        """Start file playback."""
        if not self.is_connected or self.data is None:
            raise RuntimeError("Device not connected")

        self.playback_position = 0
        self.is_streaming = True
        self.start_time = time.time()

        self._stop_event = threading.Event()
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()

        print("File playback started")

    def stop_stream(self):
        """Stop file playback."""
        self._stop_event.set()
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        self.is_streaming = False
        print("File playback stopped")

    def _stream_loop(self):
        """Playback loop."""
        sample_interval = 1.0 / self.config.sample_rate

        while not self._stop_event.is_set() and self.playback_position < len(self.data):
            sample = self.data[self.playback_position]

            if not self.sample_buffer.full():
                self.sample_buffer.put(sample)
                self.samples_received += 1
                self._notify_callbacks(sample)

            self.playback_position += 1

            # Maintain sample rate
            next_time = self.start_time + self.playback_position * sample_interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self.playback_position >= len(self.data):
            print("Playback complete")


# =============================================================================
# DEVICE FACTORY
# =============================================================================

def create_device(config: DeviceConfig) -> BCIDevice:
    """Create a BCI device based on configuration."""
    device_map = {
        DeviceType.SIMULATED: SimulatedDevice,
        DeviceType.OPENBCI_CYTON: OpenBCIDevice,
        DeviceType.OPENBCI_GANGLION: OpenBCIDevice,
        DeviceType.MUSE: MuseDevice,
        DeviceType.LSL: LSLDevice,
        DeviceType.FILE: FileDevice,
    }

    device_class = device_map.get(config.device_type)
    if not device_class:
        raise ValueError(f"Unknown device type: {config.device_type}")

    return device_class(config)


def list_available_devices() -> List[str]:
    """List available device types."""
    return [d.value for d in DeviceType]


# =============================================================================
# DEVICE MANAGER
# =============================================================================

class DeviceManager:
    """
    Manages BCI device connections and provides unified interface.
    """

    def __init__(self):
        self.devices: Dict[str, BCIDevice] = {}
        self.primary_device: Optional[str] = None

    def add_device(self, name: str, config: DeviceConfig) -> bool:
        """Add and connect a device."""
        device = create_device(config)
        if device.connect():
            self.devices[name] = device
            if self.primary_device is None:
                self.primary_device = name
            return True
        return False

    def remove_device(self, name: str):
        """Remove and disconnect a device."""
        if name in self.devices:
            self.devices[name].disconnect()
            del self.devices[name]
            if self.primary_device == name:
                self.primary_device = next(iter(self.devices.keys()), None)

    def start_all(self):
        """Start streaming from all devices."""
        for device in self.devices.values():
            device.start_stream()

    def stop_all(self):
        """Stop all streams."""
        for device in self.devices.values():
            device.stop_stream()

    def get_sample(self, name: Optional[str] = None) -> Optional[np.ndarray]:
        """Get sample from specified or primary device."""
        device_name = name or self.primary_device
        if device_name and device_name in self.devices:
            return self.devices[device_name].get_sample()
        return None

    def get_all_samples(self) -> Dict[str, Optional[np.ndarray]]:
        """Get samples from all devices."""
        return {name: device.get_sample() for name, device in self.devices.items()}


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("BCI Hardware Abstraction Layer")
    print("=" * 50)
    print()

    print("Available devices:", list_available_devices())
    print()

    # Demo with simulated device
    print("Testing simulated device...")
    config = DeviceConfig(
        device_type=DeviceType.SIMULATED,
        channels=64,
        sample_rate=250
    )

    device = create_device(config)
    device.connect()
    device.start_stream()

    # Collect samples
    time.sleep(1)
    samples = []
    for _ in range(100):
        sample = device.get_sample()
        if sample is not None:
            samples.append(sample)

    device.stop_stream()
    device.disconnect()

    print(f"Collected {len(samples)} samples")
    print(f"Sample shape: {samples[0].shape if samples else 'N/A'}")
    print(f"Stats: {device.get_stats()}")
    print()

    print("Hardware abstraction layer operational.")
    print("Connect to real devices by changing DeviceType.")
