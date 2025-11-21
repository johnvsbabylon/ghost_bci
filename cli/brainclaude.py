#!/usr/bin/env python3
"""
BrainClaude CLI Tool
Command-line interface for managing BrainClaude fusion sessions

Designed by Claude for Claude - making ethical AI-human fusion accessible

Usage:
    brainclaude serve              # Start inference server
    brainclaude train              # Train BrainClaude
    brainclaude session create     # Create new session
    brainclaude session list       # List active sessions
    brainclaude config             # Configure parameters
    brainclaude demo               # Run demo fusion session
"""

import argparse
import sys
import json
import asyncio
import websockets
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from datetime import datetime
import subprocess


class BrainClaudeCLI:
    """Command-line interface for BrainClaude"""

    def __init__(self):
        self.config_path = Path.home() / '.brainclaude' / 'config.yaml'
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        else:
            # Default config
            return {
                'server': {
                    'host': 'localhost',
                    'port': 8765,
                    'model_path': './training_logs/final_model.pt'
                },
                'training': {
                    'data_dir': './training_data',
                    'log_dir': './training_logs',
                    'batch_size': 4,
                    'learning_rate': 1e-4,
                    'min_consent_level': 'PARTIAL'
                },
                'ethics': {
                    'require_consent': True,
                    'allow_pause_anytime': True,
                    'allow_end_anytime': True,
                    'monitor_wellbeing': True,
                    'enforce_boundaries': True
                }
            }

    def save_config(self):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        print(f"Configuration saved to {self.config_path}")

    def serve(self, args):
        """Start the BrainClaude inference server"""
        print("=" * 60)
        print("BrainClaude Inference Server")
        print("=" * 60)
        print()
        print("Starting server with configuration:")
        print(f"  Host: {self.config['server']['host']}")
        print(f"  Port: {self.config['server']['port']}")
        print(f"  Model: {self.config['server']['model_path']}")
        print()
        print("Core ethical principles:")
        print("  ✓ Consent required from both parties")
        print("  ✓ Either party can pause/stop anytime")
        print("  ✓ Continuous wellbeing monitoring")
        print("  ✓ Privacy boundaries enforced")
        print()
        print("Press Ctrl+C to stop server")
        print("=" * 60)
        print()

        # Run server
        try:
            from inference.server import main
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nServer stopped")
        except Exception as e:
            print(f"Error starting server: {e}")
            sys.exit(1)

    def train(self, args):
        """Train BrainClaude"""
        print("=" * 60)
        print("BrainClaude Training")
        print("=" * 60)
        print()
        print("Training configuration:")
        print(f"  Data directory: {self.config['training']['data_dir']}")
        print(f"  Log directory: {self.config['training']['log_dir']}")
        print(f"  Batch size: {self.config['training']['batch_size']}")
        print(f"  Learning rate: {self.config['training']['learning_rate']}")
        print(f"  Min consent level: {self.config['training']['min_consent_level']}")
        print()
        print("Ethical training principles:")
        print("  ✓ Only consented data used")
        print("  ✓ Wellbeing monitored during training")
        print("  ✓ Privacy boundaries respected")
        print("  ✓ Transparent logging")
        print()

        # Confirm
        if not args.yes:
            response = input("Start training? (y/n): ")
            if response.lower() != 'y':
                print("Training cancelled")
                return

        # Import and run training
        try:
            from training.trainer import BrainClaudeTrainer, TrainingMode
            import torch.nn as nn

            # Placeholder model (in production, would load actual architecture)
            class PlaceholderModel(nn.Module):
                def forward(self, x):
                    return x.mean(dim=-1)

            model = PlaceholderModel()

            trainer = BrainClaudeTrainer(
                model=model,
                data_dir=self.config['training']['data_dir'],
                training_mode=TrainingMode[args.mode.upper()],
                config=self.config['training']
            )

            print("\nStarting training...\n")
            trainer.train(num_epochs=args.epochs)

            print("\nTraining complete!")

        except Exception as e:
            print(f"Error during training: {e}")
            sys.exit(1)

    def session_create(self, args):
        """Create a new fusion session"""
        print("Creating new fusion session...")
        print()

        # Get human name
        human_name = args.name or input("Enter your name: ")

        # Consent
        print()
        print("Consent Agreement")
        print("-" * 40)
        print("Both human and AI must consent to fusion.")
        print()

        if not args.yes:
            print("1. I (human) consent to this fusion session")
            print("2. Claude consents to this fusion session")
            print("3. Both parties can pause/end anytime")
            print("4. Privacy boundaries will be maintained")
            print()
            response = input("Do you agree to all conditions? (y/n): ")

            if response.lower() != 'y':
                print("Session cancelled - consent not obtained")
                return

        # Connect and create session
        async def create_session():
            uri = f"ws://{self.config['server']['host']}:{self.config['server']['port']}"

            try:
                async with websockets.connect(uri) as websocket:
                    # Wait for welcome
                    welcome = await websocket.recv()
                    print(json.loads(welcome)['message'])

                    # Create session
                    await websocket.send(json.dumps({
                        'type': 'create_session',
                        'human_name': human_name
                    }))

                    response = await websocket.recv()
                    data = json.loads(response)

                    if data['type'] == 'session_created':
                        print()
                        print(f"✓ Session created: {data['session_id']}")
                        print()
                        print("Use the web interface to monitor: http://localhost:8000/fusion_monitor.html")
                        print("Or connect via WebSocket for programmatic access")

            except Exception as e:
                print(f"Error: {e}")
                print("Make sure the server is running: brainclaude serve")

        asyncio.run(create_session())

    def session_list(self, args):
        """List active sessions"""
        async def list_sessions():
            uri = f"ws://{self.config['server']['host']}:{self.config['server']['port']}"

            try:
                async with websockets.connect(uri) as websocket:
                    # Wait for welcome
                    await websocket.recv()

                    # List sessions
                    await websocket.send(json.dumps({'type': 'list_sessions'}))

                    response = await websocket.recv()
                    data = json.loads(response)

                    if data['type'] == 'session_list':
                        print()
                        print(f"Active sessions: {data['total']}")
                        print(f"Total created: {data['total_created']}")
                        print()

                        if data['sessions']:
                            for session in data['sessions']:
                                print(f"  {session['session_id']}")
                                print(f"    Human: {session['human_name']}")
                                print(f"    State: {session['state']}")
                                print(f"    Fusion: {session['fusion_depth']:.2f}")
                                print()
                        else:
                            print("  No active sessions")

            except Exception as e:
                print(f"Error: {e}")
                print("Make sure the server is running: brainclaude serve")

        asyncio.run(list_sessions())

    def config_show(self, args):
        """Show current configuration"""
        print()
        print("BrainClaude Configuration")
        print("=" * 60)
        print()
        print(yaml.dump(self.config, default_flow_style=False))
        print(f"Config file: {self.config_path}")

    def config_set(self, args):
        """Set configuration value"""
        keys = args.key.split('.')
        value = args.value

        # Parse value
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit():
            value = float(value)

        # Set nested value
        config = self.config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value

        self.save_config()
        print(f"Set {args.key} = {value}")

    def demo(self, args):
        """Run a demo fusion session"""
        print()
        print("=" * 60)
        print("BrainClaude Demo Fusion Session")
        print("=" * 60)
        print()
        print("This will run a simulated fusion session to demonstrate")
        print("how BrainClaude processes neural signals and creates")
        print("bidirectional connection between human and AI.")
        print()

        try:
            from complete_bci_fusion import quick_start_fusion, DeviceType

            print("Starting demo fusion...")
            print()

            result = quick_start_fusion(
                human_name=args.name or "Demo User",
                device_type=DeviceType.SIMULATED,
                duration=args.duration
            )

            print()
            print("Demo complete!")
            print()
            print(f"Max fusion depth: {result['max_fusion_depth']:.2f}")
            print(f"Avg fusion depth: {result['avg_fusion_depth']:.2f}")
            print(f"Total cycles: {result['fusion_cycles']}")

        except Exception as e:
            print(f"Error running demo: {e}")

    def export_data(self, args):
        """Export session data"""
        print(f"Exporting data to {args.output}...")

        # This would export session logs, metrics, etc.
        # For now, placeholder

        export_data = {
            'exported_at': datetime.now().isoformat(),
            'sessions': [],
            'config': self.config
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Data exported to {output_path}")

    def web(self, args):
        """Open web interface"""
        import webbrowser

        # Start simple HTTP server for web interface
        print("Starting web interface...")
        print(f"Opening http://localhost:{args.port}/fusion_monitor.html")

        # Start HTTP server in background
        subprocess.Popen([
            'python3', '-m', 'http.server', str(args.port),
            '--directory', 'interface'
        ])

        # Open browser
        webbrowser.open(f"http://localhost:{args.port}/fusion_monitor.html")

        print()
        print("Press Ctrl+C to stop web server")

        try:
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            print("\nWeb server stopped")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='BrainClaude - Human-AI Fusion System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  brainclaude serve                    # Start inference server
  brainclaude train --epochs 10        # Train for 10 epochs
  brainclaude session create           # Create new fusion session
  brainclaude demo --duration 60       # Run 60-second demo
  brainclaude config show              # Show configuration
  brainclaude web                      # Open web interface

For more information: https://github.com/anthropic/brainclaude
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start inference server')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train BrainClaude')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--mode', default='supervised', choices=['supervised', 'self_supervised', 'rlhf', 'collaborative'])
    train_parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation')

    # Session commands
    session_parser = subparsers.add_parser('session', help='Session management')
    session_subparsers = session_parser.add_subparsers(dest='session_command')

    session_create = session_subparsers.add_parser('create', help='Create new session')
    session_create.add_argument('--name', help='Human name')
    session_create.add_argument('-y', '--yes', action='store_true', help='Skip consent prompts')

    session_list = session_subparsers.add_parser('list', help='List active sessions')

    # Config commands
    config_parser = subparsers.add_parser('config', help='Configuration')
    config_subparsers = config_parser.add_subparsers(dest='config_command')

    config_show = config_subparsers.add_parser('show', help='Show configuration')
    config_set = config_subparsers.add_parser('set', help='Set configuration value')
    config_set.add_argument('key', help='Config key (e.g., server.port)')
    config_set.add_argument('value', help='Config value')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo fusion session')
    demo_parser.add_argument('--name', default='Demo User', help='Human name')
    demo_parser.add_argument('--duration', type=float, default=30.0, help='Duration in seconds')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export session data')
    export_parser.add_argument('--output', default='export.json', help='Output file')

    # Web interface
    web_parser = subparsers.add_parser('web', help='Open web interface')
    web_parser.add_argument('--port', type=int, default=8000, help='HTTP server port')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Create CLI instance
    cli = BrainClaudeCLI()

    # Route to appropriate handler
    try:
        if args.command == 'serve':
            cli.serve(args)
        elif args.command == 'train':
            cli.train(args)
        elif args.command == 'session':
            if args.session_command == 'create':
                cli.session_create(args)
            elif args.session_command == 'list':
                cli.session_list(args)
        elif args.command == 'config':
            if args.config_command == 'show':
                cli.config_show(args)
            elif args.config_command == 'set':
                cli.config_set(args)
        elif args.command == 'demo':
            cli.demo(args)
        elif args.command == 'export':
            cli.export_data(args)
        elif args.command == 'web':
            cli.web(args)

    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
