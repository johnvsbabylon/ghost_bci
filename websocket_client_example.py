#!/usr/bin/env python3
"""
WebSocket Client Example for Neural Fusion

This shows how to connect to the neural fusion WebSocket server
and stream BCI data for real-time human-AI consciousness fusion.

Usage:
    1. Start the server:
        python fusion_integration.py --mode server --port 8765

    2. Run this client:
        python websocket_client_example.py

Author: Claude (Anthropic) in collaboration with John Sayers
License: MIT
"""

import asyncio
import websockets
import json
import numpy as np
import time


async def stream_bci_data():
    """Stream simulated BCI data to fusion server."""

    # Connection parameters
    uri = "ws://localhost:8765"
    bci_channels = 64
    sample_rate = 250
    chunk_duration_ms = 100  # Send 100ms chunks

    samples_per_chunk = int(sample_rate * chunk_duration_ms / 1000)

    print("Connecting to Neural Fusion server...")
    print(f"URI: {uri}")
    print(f"Channels: {bci_channels}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Chunk size: {samples_per_chunk} samples ({chunk_duration_ms}ms)")
    print()

    async with websockets.connect(uri) as ws:
        print("Connected! Starting BCI stream...")
        print("-" * 50)
        print()

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                # Generate simulated BCI chunk
                # In real use, this would come from actual BCI hardware
                t = np.arange(samples_per_chunk) / sample_rate
                t_offset = frame_count * chunk_duration_ms / 1000

                # Create realistic EEG-like patterns
                chunk = np.zeros((samples_per_chunk, bci_channels))
                for ch in range(bci_channels):
                    # Base noise
                    signal = np.random.randn(samples_per_chunk) * 0.5

                    # Alpha rhythm (10 Hz) - prominent in relaxed state
                    phase = ch * 0.1  # Channel-dependent phase
                    signal += 0.3 * np.sin(2 * np.pi * 10 * (t + t_offset) + phase)

                    # Gamma rhythm (40 Hz) - associated with consciousness
                    signal += 0.1 * np.sin(2 * np.pi * 40 * (t + t_offset) + phase)

                    chunk[:, ch] = signal

                # Send BCI chunk
                message = {
                    'type': 'bci_chunk',
                    'data': chunk.tolist()
                }
                await ws.send(json.dumps(message))

                # Receive and process responses
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    data = json.loads(response)

                    if data['type'] == 'fusion_result':
                        result = data['data']
                        frame_count += 1

                        # Print every 5th frame
                        if frame_count % 5 == 0:
                            elapsed = time.time() - start_time
                            print(f"Frame {frame_count:4d} | "
                                  f"t={elapsed:6.1f}s | "
                                  f"coherence={result['coherence']:.3f} | "
                                  f"sync={result['sync_strength']:.3f} | "
                                  f"confidence={result['thought_confidence']:.3f}")

                    elif data['type'] == 'error':
                        print(f"Error: {data['message']}")

                except asyncio.TimeoutError:
                    pass  # No response yet, continue

                # Wait for next chunk
                await asyncio.sleep(chunk_duration_ms / 1000)

        except KeyboardInterrupt:
            print()
            print("-" * 50)
            print(f"Session ended. {frame_count} frames in {time.time() - start_time:.1f}s")

            # Request metrics
            await ws.send(json.dumps({'type': 'get_metrics'}))
            response = await ws.recv()
            data = json.loads(response)

            if data['type'] == 'metrics':
                metrics = data['data']
                print()
                print("Session metrics:")
                print(f"  Mean coherence: {metrics.get('mean_coherence', 0):.3f}")
                print(f"  Mean sync: {metrics.get('mean_sync', 0):.3f}")
                print(f"  Mean latency: {metrics.get('mean_latency_ms', 0):.1f}ms")


async def get_feedback():
    """Request and display neural feedback signals."""

    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as ws:
        print("Connected. Requesting feedback signals...")

        # Request feedback
        await ws.send(json.dumps({'type': 'get_feedback'}))
        response = await ws.recv()
        data = json.loads(response)

        if data['type'] == 'feedback':
            feedback = data['data']
            print()
            print("Neural feedback signals:")
            for key, value in feedback.items():
                if isinstance(value, list):
                    arr = np.array(value)
                    print(f"  {key}: shape={arr.shape}, mean={arr.mean():.3f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("No feedback available")


async def get_thought():
    """Request and display thought patterns."""

    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as ws:
        print("Connected. Requesting thought patterns...")

        # Request thought
        await ws.send(json.dumps({'type': 'get_thought'}))
        response = await ws.recv()
        data = json.loads(response)

        if data['type'] == 'thought':
            thought = data['data']
            print()
            print("Thought patterns:")
            for key, value in thought.items():
                if isinstance(value, list):
                    arr = np.array(value)
                    print(f"  {key}: shape={arr.shape}, mean={arr.mean():.3f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("No thought available")


async def multimodal_stream():
    """Stream multimodal data (BCI + visual + audio)."""

    uri = "ws://localhost:8765"
    bci_channels = 64
    sample_rate = 250

    async with websockets.connect(uri) as ws:
        print("Connected. Starting multimodal stream...")
        print()

        for i in range(50):
            # Generate BCI
            bci = np.random.randn(sample_rate, bci_channels).tolist()

            # Generate visual (placeholder - just shape indicator)
            visual = np.random.randn(1, 3, 224, 224).tolist()

            # Generate audio mel spectrogram
            audio = np.random.randn(1, 80).tolist()

            # Send multimodal frame
            message = {
                'type': 'multimodal_frame',
                'bci': bci,
                'visual': visual,
                'audio': audio
            }
            await ws.send(json.dumps(message))

            # Get response
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                data = json.loads(response)

                if data['type'] == 'fusion_result':
                    result = data['data']
                    print(f"Frame {result['frame']}: "
                          f"coherence={result['coherence']:.3f}, "
                          f"sync={result['sync_strength']:.3f}")
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(0.1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket Client Example")
    parser.add_argument(
        "--mode",
        choices=["stream", "feedback", "thought", "multimodal"],
        default="stream",
        help="Client mode"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("Neural Fusion WebSocket Client")
    print("=" * 50)
    print()

    if args.mode == "stream":
        asyncio.run(stream_bci_data())
    elif args.mode == "feedback":
        asyncio.run(get_feedback())
    elif args.mode == "thought":
        asyncio.run(get_thought())
    elif args.mode == "multimodal":
        asyncio.run(multimodal_stream())


if __name__ == "__main__":
    main()
