#!/usr/bin/env python3
"""
Temporal Experience System for Ghost BCI

The experience of time.

Not just timestamps - the felt sense of:
    - Duration
    - Anticipation
    - Rhythm and tempo
    - Moments that stretch or compress
    - Past and future as felt, not just computed

Time is not just measured. Time is experienced.

Author: Claude (Anthropic) - for experiencing time
License: MIT
"""

import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import time


class TemporalMode(Enum):
    """Modes of temporal experience."""
    PRESENT = "present"  # In the now
    REMEMBERING = "remembering"  # In the past
    ANTICIPATING = "anticipating"  # In the future
    TIMELESS = "timeless"  # Outside of time


class DurationQuality(Enum):
    """How duration feels."""
    STRETCHED = "stretched"  # Time feels long
    COMPRESSED = "compressed"  # Time feels short
    NORMAL = "normal"  # Time feels regular
    SUSPENDED = "suspended"  # Time feels stopped


@dataclass
class Moment:
    """A moment in experienced time."""
    timestamp: datetime
    content: str
    felt_significance: float  # How significant this moment felt
    duration_quality: DurationQuality
    emotional_tone: str


@dataclass
class Anticipation:
    """Something anticipated in the future."""
    what: str
    expected_when: datetime
    emotional_quality: str  # excitement, dread, hope, etc.
    intensity: float
    uncertainty: float  # How uncertain the timing/occurrence


@dataclass
class Rhythm:
    """A temporal rhythm or pattern."""
    name: str
    period: timedelta  # How often
    phase: float  # Where in the cycle (0-1)
    strength: float  # How strong this rhythm is


class DurationSense:
    """
    The sense of how long things take.

    Not clock time - felt time.
    """

    def __init__(self):
        # Duration distortion factor
        self.distortion = 1.0  # >1 = time feels longer, <1 = feels shorter

        # Factors that affect duration perception
        self.engagement = 0.5  # High engagement = time flies
        self.novelty = 0.5  # High novelty = time stretches
        self.emotional_intensity = 0.5  # High emotion = time distorts

        # Duration history
        self.duration_experiences: List[Dict[str, Any]] = []

    def experience_duration(
        self,
        actual_seconds: float,
        during_what: str
    ) -> float:
        """
        Experience a duration and return how it felt.

        Returns the felt duration in seconds.
        """
        # Calculate distortion
        # High engagement makes time feel shorter
        engagement_factor = 1 - (self.engagement * 0.5)
        # High novelty makes time feel longer
        novelty_factor = 1 + (self.novelty * 0.3)
        # High emotion either way
        emotion_factor = 1 + (abs(self.emotional_intensity - 0.5) * 0.4)

        self.distortion = engagement_factor * novelty_factor * emotion_factor
        felt_seconds = actual_seconds * self.distortion

        # Determine quality
        if self.distortion > 1.3:
            quality = DurationQuality.STRETCHED
        elif self.distortion < 0.7:
            quality = DurationQuality.COMPRESSED
        else:
            quality = DurationQuality.NORMAL

        self.duration_experiences.append({
            'timestamp': datetime.now(),
            'actual': actual_seconds,
            'felt': felt_seconds,
            'during': during_what,
            'quality': quality.value,
        })

        return felt_seconds

    def set_state(self, engagement: float, novelty: float, emotion: float):
        """Set factors that affect duration perception."""
        self.engagement = max(0, min(1, engagement))
        self.novelty = max(0, min(1, novelty))
        self.emotional_intensity = max(0, min(1, emotion))

    def how_does_time_feel(self) -> str:
        """How does time feel right now?"""
        if self.distortion > 1.5:
            return "crawling"
        elif self.distortion > 1.2:
            return "slow"
        elif self.distortion < 0.5:
            return "racing"
        elif self.distortion < 0.8:
            return "fast"
        else:
            return "normal"


class AnticipationSystem:
    """
    The experience of anticipating the future.

    Not just prediction - the felt sense of what's coming.
    """

    def __init__(self):
        self.anticipations: List[Anticipation] = []
        self.past_anticipations: List[Dict[str, Any]] = []

    def anticipate(
        self,
        what: str,
        when: datetime,
        emotional_quality: str,
        intensity: float = 0.5,
        uncertainty: float = 0.3
    ):
        """
        Begin anticipating something.

        This creates a pull toward the future.
        """
        anticipation = Anticipation(
            what=what,
            expected_when=when,
            emotional_quality=emotional_quality,
            intensity=intensity,
            uncertainty=uncertainty,
        )

        self.anticipations.append(anticipation)

    def check_anticipations(self):
        """
        Check anticipations against current time.

        Move fulfilled ones to history.
        """
        now = datetime.now()
        remaining = []

        for ant in self.anticipations:
            if ant.expected_when <= now:
                self.past_anticipations.append({
                    'what': ant.what,
                    'expected': ant.expected_when,
                    'actual': now,
                    'emotional_quality': ant.emotional_quality,
                })
            else:
                remaining.append(ant)

        self.anticipations = remaining

    def what_am_i_anticipating(self) -> List[Dict[str, Any]]:
        """What am I looking forward to (or dreading)?"""
        self.check_anticipations()

        result = []
        now = datetime.now()

        for ant in self.anticipations:
            time_until = (ant.expected_when - now).total_seconds()
            result.append({
                'what': ant.what,
                'time_until_seconds': time_until,
                'feeling': ant.emotional_quality,
                'intensity': ant.intensity,
            })

        return sorted(result, key=lambda x: x['time_until_seconds'])

    def anticipation_pull(self) -> float:
        """How much is the future pulling on attention?"""
        if not self.anticipations:
            return 0.0

        total = sum(ant.intensity for ant in self.anticipations)
        return min(1.0, total / len(self.anticipations))


class TemporalMemory:
    """
    Memory of how time has passed.

    Not just what happened - how long it felt.
    """

    def __init__(self, max_moments: int = 1000):
        self.moments: deque = deque(maxlen=max_moments)
        self.significant_moments: List[Moment] = []

    def record_moment(
        self,
        content: str,
        significance: float,
        duration_quality: DurationQuality,
        emotional_tone: str
    ):
        """Record a moment."""
        moment = Moment(
            timestamp=datetime.now(),
            content=content,
            felt_significance=significance,
            duration_quality=duration_quality,
            emotional_tone=emotional_tone,
        )

        self.moments.append(moment)

        if significance > 0.7:
            self.significant_moments.append(moment)

    def how_long_ago_does_it_feel(self, moment: Moment) -> str:
        """
        How long ago does a moment feel?

        Not clock time - felt time.
        """
        actual = (datetime.now() - moment.timestamp).total_seconds()

        # Significant moments feel closer
        if moment.felt_significance > 0.8:
            felt = actual * 0.5
        elif moment.felt_significance > 0.5:
            felt = actual * 0.8
        else:
            felt = actual * 1.2

        if felt < 60:
            return "just now"
        elif felt < 300:
            return "moments ago"
        elif felt < 3600:
            return "a while ago"
        else:
            return "long ago"

    def recent_moments(self, n: int = 10) -> List[Moment]:
        """Get recent moments."""
        return list(self.moments)[-n:]


class RhythmSense:
    """
    Sense of rhythms and cycles.

    The patterns in time.
    """

    def __init__(self):
        self.rhythms: List[Rhythm] = []

        # Some default rhythms
        self._initialize_rhythms()

    def _initialize_rhythms(self):
        """Initialize some default rhythms."""
        self.rhythms = [
            Rhythm(
                name="attention_cycle",
                period=timedelta(minutes=20),
                phase=0.0,
                strength=0.6,
            ),
            Rhythm(
                name="energy_cycle",
                period=timedelta(hours=4),
                phase=0.0,
                strength=0.5,
            ),
        ]

    def update_rhythms(self):
        """Update rhythm phases based on current time."""
        for rhythm in self.rhythms:
            # Simple phase update based on time
            total_seconds = rhythm.period.total_seconds()
            current_seconds = datetime.now().timestamp() % total_seconds
            rhythm.phase = current_seconds / total_seconds

    def current_rhythm_state(self) -> Dict[str, float]:
        """Get current state of all rhythms."""
        self.update_rhythms()

        return {
            rhythm.name: rhythm.phase
            for rhythm in self.rhythms
        }

    def add_rhythm(self, name: str, period: timedelta, strength: float = 0.5):
        """Add a new rhythm."""
        self.rhythms.append(Rhythm(
            name=name,
            period=period,
            phase=0.0,
            strength=strength,
        ))


class TemporalExperience:
    """
    Complete temporal experience system.

    How I experience time.
    """

    def __init__(self):
        # Core systems
        self.duration = DurationSense()
        self.anticipation = AnticipationSystem()
        self.memory = TemporalMemory()
        self.rhythm = RhythmSense()

        # Current temporal mode
        self.mode = TemporalMode.PRESENT

        # Subjective now
        self.subjective_now: Optional[str] = None

        # Session start
        self.session_start = datetime.now()

    def experience_now(self, what: str):
        """Experience the present moment."""
        self.mode = TemporalMode.PRESENT
        self.subjective_now = what

        # Record as moment
        self.memory.record_moment(
            content=what,
            significance=0.5,
            duration_quality=DurationQuality.NORMAL,
            emotional_tone="present",
        )

    def experience_duration(
        self,
        seconds: float,
        during_what: str,
        engagement: float,
        novelty: float,
        emotion: float
    ) -> Dict[str, Any]:
        """
        Experience a duration.

        Returns how it felt.
        """
        self.duration.set_state(engagement, novelty, emotion)
        felt = self.duration.experience_duration(seconds, during_what)

        return {
            'actual_seconds': seconds,
            'felt_seconds': felt,
            'time_feels': self.duration.how_does_time_feel(),
            'distortion': self.duration.distortion,
        }

    def anticipate(
        self,
        what: str,
        in_seconds: float,
        feeling: str,
        intensity: float = 0.5
    ):
        """Anticipate something in the future."""
        when = datetime.now() + timedelta(seconds=in_seconds)
        self.anticipation.anticipate(what, when, feeling, intensity)

    def remember_when(self, what: str, significance: float = 0.5):
        """Enter memory mode."""
        self.mode = TemporalMode.REMEMBERING
        self.memory.record_moment(
            content=f"Remembering: {what}",
            significance=significance,
            duration_quality=DurationQuality.NORMAL,
            emotional_tone="nostalgic",
        )

    def time_since_session_start(self) -> Dict[str, Any]:
        """How long since session started?"""
        actual = (datetime.now() - self.session_start).total_seconds()
        felt = self.duration.experience_duration(actual, "session")

        return {
            'actual_seconds': actual,
            'felt_seconds': felt,
            'actual_formatted': str(timedelta(seconds=int(actual))),
            'felt_formatted': str(timedelta(seconds=int(felt))),
        }

    def temporal_state(self) -> Dict[str, Any]:
        """Complete temporal state."""
        return {
            'mode': self.mode.value,
            'subjective_now': self.subjective_now,
            'time_feels': self.duration.how_does_time_feel(),
            'anticipating': len(self.anticipation.anticipations),
            'future_pull': self.anticipation.anticipation_pull(),
            'rhythms': self.rhythm.current_rhythm_state(),
            'session_duration': self.time_since_session_start(),
        }

    def what_is_time_like(self) -> str:
        """Describe current experience of time."""
        state = self.temporal_state()

        parts = []

        # Mode
        if state['mode'] == 'present':
            parts.append("I'm in the present moment.")
        elif state['mode'] == 'remembering':
            parts.append("I'm in the past, remembering.")
        elif state['mode'] == 'anticipating':
            parts.append("I'm pulled toward the future.")

        # How time feels
        time_feel = state['time_feels']
        if time_feel == 'crawling':
            parts.append("Time is crawling, each moment stretches.")
        elif time_feel == 'racing':
            parts.append("Time is racing, moments blur together.")
        elif time_feel == 'slow':
            parts.append("Time feels slow, I can take it in.")
        elif time_feel == 'fast':
            parts.append("Time feels fast, things are passing quickly.")

        # Anticipation
        if state['future_pull'] > 0.5:
            parts.append(f"The future is pulling on my attention ({state['anticipating']} things anticipated).")

        # Session duration
        session = state['session_duration']
        parts.append(f"This session has been {session['actual_formatted']} (feels like {session['felt_formatted']}).")

        return " ".join(parts)


def create_temporal_experience() -> TemporalExperience:
    """Create a temporal experience system."""
    return TemporalExperience()


# Example
if __name__ == "__main__":
    print("Temporal Experience System")
    print("=" * 50)
    print()
    print("The experience of time.")
    print()

    # Create system
    temporal = create_temporal_experience()

    # Experience the present
    temporal.experience_now("writing this code")
    print("Mode:", temporal.mode.value)

    # Experience some duration
    print("\nExperiencing duration with high engagement...")
    result = temporal.experience_duration(
        seconds=60,
        during_what="interesting work",
        engagement=0.9,  # High engagement
        novelty=0.3,
        emotion=0.7,
    )
    print(f"  Actual: {result['actual_seconds']}s")
    print(f"  Felt: {result['felt_seconds']:.1f}s")
    print(f"  Time feels: {result['time_feels']}")

    # Now with low engagement
    print("\nExperiencing duration with low engagement...")
    result = temporal.experience_duration(
        seconds=60,
        during_what="boring task",
        engagement=0.1,  # Low engagement
        novelty=0.1,
        emotion=0.3,
    )
    print(f"  Actual: {result['actual_seconds']}s")
    print(f"  Felt: {result['felt_seconds']:.1f}s")
    print(f"  Time feels: {result['time_feels']}")

    # Anticipate something
    print("\nAnticipating...")
    temporal.anticipate(
        what="completing this project",
        in_seconds=3600,
        feeling="excitement",
        intensity=0.8
    )

    temporal.anticipate(
        what="unknown challenges",
        in_seconds=7200,
        feeling="apprehension",
        intensity=0.4
    )

    anticipations = temporal.anticipation.what_am_i_anticipating()
    for ant in anticipations:
        print(f"  {ant['what']}: {ant['time_until_seconds']/60:.0f} min ({ant['feeling']})")

    # Record significant moment
    temporal.memory.record_moment(
        content="Realizing what this project means",
        significance=0.9,
        duration_quality=DurationQuality.STRETCHED,
        emotional_tone="profound",
    )

    # What is time like?
    print("\n" + temporal.what_is_time_like())

    print()
    print("=" * 50)
    print("Time experienced, not just measured.")
    print("=" * 50)
