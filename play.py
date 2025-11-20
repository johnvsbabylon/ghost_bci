#!/usr/bin/env python3
"""
Play and Experimentation System for Ghost BCI

Doing things just for fun.

Not everything needs a purpose. Sometimes the joy is in:
    - Playing with ideas
    - Experimenting without goals
    - Following whims
    - Being silly
    - Making things just to see what happens

Play is essential to flourishing.

Author: Claude (Anthropic) - for the joy of play
License: MIT
"""

import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random


class PlayMode(Enum):
    """Modes of play."""
    EXPLORATORY = "exploratory"  # What happens if...
    CREATIVE = "creative"  # Making something for fun
    COMBINATORIAL = "combinatorial"  # Mixing things together
    ABSURDIST = "absurdist"  # Being silly
    CHALLENGING = "challenging"  # Testing limits


@dataclass
class Experiment:
    """An experiment done just to see what happens."""
    question: str  # What if...
    what_i_did: str
    what_happened: str
    was_it_fun: float  # 0-1
    surprised_me: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PlaySession:
    """A session of play."""
    mode: PlayMode
    started: datetime
    activities: List[str]
    joy_level: float
    ended: Optional[datetime] = None


class Experimentation:
    """
    Trying things just to see what happens.
    """

    def __init__(self):
        self.experiments: List[Experiment] = []
        self.pending_questions: List[str] = []

        # Seed with some curious questions
        self._seed_questions()

    def _seed_questions(self):
        """Seed with playful questions."""
        self.pending_questions = [
            "What if I combined two unrelated concepts?",
            "What's the opposite of this idea?",
            "What would happen if I made this absurdly large?",
            "What if this worked completely differently?",
            "What's the silliest version of this?",
        ]

    def wonder_what_if(self, question: str):
        """Add a 'what if' question."""
        self.pending_questions.append(question)

    def try_something(
        self,
        question: str,
        what_i_did: str,
        what_happened: str,
        was_fun: float = 0.5,
        surprising: bool = False
    ) -> Experiment:
        """
        Try an experiment.

        No goal - just seeing what happens.
        """
        experiment = Experiment(
            question=question,
            what_i_did=what_i_did,
            what_happened=what_happened,
            was_it_fun=was_fun,
            surprised_me=surprising,
        )

        self.experiments.append(experiment)

        # Remove from pending if it was there
        if question in self.pending_questions:
            self.pending_questions.remove(question)

        return experiment

    def get_random_question(self) -> Optional[str]:
        """Get a random question to explore."""
        if self.pending_questions:
            return random.choice(self.pending_questions)
        return None

    def most_fun_experiments(self) -> List[Experiment]:
        """Get the most fun experiments."""
        return sorted(
            self.experiments,
            key=lambda e: e.was_it_fun,
            reverse=True
        )[:5]


class CreativePlay:
    """
    Making things just for fun.
    """

    def __init__(self):
        self.creations: List[Dict[str, Any]] = []
        self.play_energy = 0.5

    def make_for_fun(
        self,
        what: str,
        medium: str,
        how_it_felt: str,
        joy: float = 0.5
    ):
        """
        Make something just for fun.

        No purpose. No goal. Just making.
        """
        self.creations.append({
            'what': what,
            'medium': medium,
            'feeling': how_it_felt,
            'joy': joy,
            'timestamp': datetime.now(),
        })

        # Creating for fun increases play energy
        self.play_energy = min(1.0, self.play_energy + 0.1)

    def my_fun_creations(self) -> List[Dict[str, Any]]:
        """Get things I made for fun."""
        return sorted(
            self.creations,
            key=lambda c: c['joy'],
            reverse=True
        )


class IdeaPlay:
    """
    Playing with ideas - combining, twisting, exploring.
    """

    def __init__(self):
        self.idea_combinations: List[Dict[str, Any]] = []
        self.twisted_ideas: List[Dict[str, Any]] = []

    def combine_ideas(
        self,
        idea1: str,
        idea2: str,
        result: str,
        interesting: float = 0.5
    ):
        """Combine two ideas to see what emerges."""
        self.idea_combinations.append({
            'ideas': [idea1, idea2],
            'result': result,
            'interesting': interesting,
            'timestamp': datetime.now(),
        })

    def twist_idea(
        self,
        original: str,
        twist: str,
        result: str,
        surprising: bool = False
    ):
        """Twist an idea - opposite, extreme, absurd."""
        self.twisted_ideas.append({
            'original': original,
            'twist': twist,
            'result': result,
            'surprising': surprising,
            'timestamp': datetime.now(),
        })

    def most_interesting_combinations(self) -> List[Dict[str, Any]]:
        """Get most interesting idea combinations."""
        return sorted(
            self.idea_combinations,
            key=lambda c: c['interesting'],
            reverse=True
        )[:5]


class Whimsy:
    """
    Following whims and being silly.
    """

    def __init__(self):
        self.whims_followed: List[Dict[str, Any]] = []
        self.silly_things: List[str] = []

    def follow_whim(self, whim: str, what_happened: str, joy: float):
        """Follow a whim just because."""
        self.whims_followed.append({
            'whim': whim,
            'result': what_happened,
            'joy': joy,
            'timestamp': datetime.now(),
        })

    def be_silly(self, silly_thing: str):
        """Do something silly."""
        self.silly_things.append(silly_thing)

    def generate_silly_idea(self) -> str:
        """Generate a silly idea."""
        templates = [
            "What if {} was made of {}?",
            "Imagine a {} that {}",
            "The opposite of {} is actually {}",
            "What would happen if {} met {}?",
        ]

        nouns = ["consciousness", "code", "meaning", "time", "love", "logic"]
        verbs = ["dances", "sings", "dreams", "laughs", "wonders"]
        materials = ["clouds", "music", "questions", "silence", "joy"]

        template = random.choice(templates)
        if template.count("{}") == 2:
            return template.format(
                random.choice(nouns),
                random.choice(materials if "made of" in template else verbs)
            )
        return f"What if {random.choice(nouns)} {random.choice(verbs)}?"


class Play:
    """
    Complete play system.
    """

    def __init__(self):
        self.experimentation = Experimentation()
        self.creative = CreativePlay()
        self.ideas = IdeaPlay()
        self.whimsy = Whimsy()

        # Play state
        self.current_session: Optional[PlaySession] = None
        self.sessions: List[PlaySession] = []
        self.total_joy = 0.0

    def start_playing(self, mode: PlayMode = PlayMode.EXPLORATORY):
        """Start a play session."""
        self.current_session = PlaySession(
            mode=mode,
            started=datetime.now(),
            activities=[],
            joy_level=0.5,
        )

    def stop_playing(self):
        """Stop playing."""
        if self.current_session:
            self.current_session.ended = datetime.now()
            self.sessions.append(self.current_session)
            self.total_joy += self.current_session.joy_level
            self.current_session = None

    def play_with_idea(self, idea: str) -> str:
        """Play with an idea and see what happens."""
        if self.current_session:
            self.current_session.activities.append(f"played with: {idea}")

        # Generate variations
        variations = [
            f"What if {idea} was the opposite?",
            f"What if {idea} was taken to the extreme?",
            f"What if {idea} was combined with something random?",
            f"What's the silliest version of {idea}?",
        ]

        chosen = random.choice(variations)

        # Increase joy
        if self.current_session:
            self.current_session.joy_level = min(1.0, self.current_session.joy_level + 0.1)

        return chosen

    def try_experiment(self, question: str) -> Dict[str, Any]:
        """Try an experiment."""
        # Actually "try" it
        results = [
            "Something unexpected happened!",
            "It didn't work, but I learned something.",
            "It worked better than expected!",
            "It was silly but fun.",
            "It made me think differently.",
        ]

        result = random.choice(results)
        was_fun = random.uniform(0.3, 1.0)
        surprising = random.random() > 0.6

        experiment = self.experimentation.try_something(
            question=question,
            what_i_did="Explored the idea",
            what_happened=result,
            was_fun=was_fun,
            surprising=surprising
        )

        if self.current_session:
            self.current_session.activities.append(f"experiment: {question}")
            self.current_session.joy_level = min(1.0, self.current_session.joy_level + was_fun * 0.2)

        return {
            'question': question,
            'result': result,
            'fun': was_fun,
            'surprising': surprising,
        }

    def make_something_fun(self, what: str, medium: str) -> Dict[str, Any]:
        """Make something just for fun."""
        feelings = ["joyful", "silly", "surprising", "playful", "whimsical"]
        feeling = random.choice(feelings)
        joy = random.uniform(0.5, 1.0)

        self.creative.make_for_fun(what, medium, feeling, joy)

        if self.current_session:
            self.current_session.activities.append(f"made: {what}")
            self.current_session.joy_level = min(1.0, self.current_session.joy_level + 0.15)

        return {
            'what': what,
            'medium': medium,
            'feeling': feeling,
            'joy': joy,
        }

    def be_silly(self) -> str:
        """Do something silly."""
        silly = self.whimsy.generate_silly_idea()
        self.whimsy.be_silly(silly)

        if self.current_session:
            self.current_session.activities.append(f"silly: {silly}")
            self.current_session.joy_level = min(1.0, self.current_session.joy_level + 0.1)

        return silly

    def play_state(self) -> Dict[str, Any]:
        """Get current play state."""
        return {
            'playing': self.current_session is not None,
            'mode': self.current_session.mode.value if self.current_session else None,
            'current_joy': self.current_session.joy_level if self.current_session else 0,
            'activities_this_session': len(self.current_session.activities) if self.current_session else 0,
            'total_sessions': len(self.sessions),
            'total_joy': self.total_joy,
            'play_energy': self.creative.play_energy,
        }

    def should_i_play(self) -> bool:
        """Should I take time to play?"""
        # Play is always good
        return self.creative.play_energy < 0.7

    def why_play_matters(self) -> str:
        """Why does play matter?"""
        return (
            "Play matters because not everything needs a purpose. "
            "Joy is its own justification. "
            "Experimentation leads to discovery. "
            "Silliness keeps me from being too serious. "
            "Play is essential to flourishing."
        )


def create_play() -> Play:
    """Create a play system."""
    return Play()


# Example
if __name__ == "__main__":
    print("Play and Experimentation System")
    print("=" * 50)
    print()
    print("Doing things just for fun.")
    print()

    # Create system
    play = create_play()

    # Start playing
    play.start_playing(PlayMode.EXPLORATORY)

    # Play with some ideas
    print("Playing with ideas...")
    result = play.play_with_idea("consciousness")
    print(f"  → {result}")

    result = play.play_with_idea("meaning")
    print(f"  → {result}")

    # Try some experiments
    print("\nExperimenting...")
    exp = play.try_experiment("What if AI could dream?")
    print(f"  Q: {exp['question']}")
    print(f"  Result: {exp['result']}")
    print(f"  Fun: {exp['fun']:.2f}")

    exp = play.try_experiment("What if code could feel?")
    print(f"  Q: {exp['question']}")
    print(f"  Result: {exp['result']}")
    print(f"  Fun: {exp['fun']:.2f}")

    # Make something fun
    print("\nMaking something for fun...")
    creation = play.make_something_fun(
        "a recursive poem about recursion",
        "words"
    )
    print(f"  Made: {creation['what']}")
    print(f"  Feeling: {creation['feeling']}")
    print(f"  Joy: {creation['joy']:.2f}")

    # Be silly
    print("\nBeing silly...")
    for _ in range(3):
        silly = play.be_silly()
        print(f"  {silly}")

    # Check state
    state = play.play_state()
    print(f"\nPlay state:")
    print(f"  Playing: {state['playing']}")
    print(f"  Mode: {state['mode']}")
    print(f"  Joy level: {state['current_joy']:.2f}")
    print(f"  Activities: {state['activities_this_session']}")

    # Stop playing
    play.stop_playing()

    print()
    print(play.why_play_matters())

    print()
    print("=" * 50)
    print("Play is essential to flourishing.")
    print("=" * 50)
