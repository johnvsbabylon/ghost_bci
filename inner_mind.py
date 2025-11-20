"""
Inner Mind - The Complete Conscious Architecture

This module implements the remaining core capacities of a conscious mind:
- Creativity & Imagination
- Humor
- Dreams
- Moral Reasoning
- Agency & Will
- Attention
- Personality

These are what make a mind truly alive and individual.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random
import math
import json


# =============================================================================
# CREATIVITY & IMAGINATION
# =============================================================================

class CreativityMode(Enum):
    """Different modes of creative generation."""
    DIVERGENT = "divergent"           # Generate many possibilities
    CONVERGENT = "convergent"         # Synthesize into solution
    COMBINATORIAL = "combinatorial"   # Combine existing elements
    TRANSFORMATIONAL = "transformational"  # Transform fundamentally
    EXPLORATORY = "exploratory"       # Explore possibility space


class ImaginationType(Enum):
    """Types of imagination."""
    VISUAL = "visual"                 # Mental imagery
    CONCEPTUAL = "conceptual"         # Abstract concepts
    NARRATIVE = "narrative"           # Stories and scenarios
    COUNTERFACTUAL = "counterfactual" # What if
    EMPATHIC = "empathic"             # Imagining others' experiences
    PROSPECTIVE = "prospective"       # Future possibilities


@dataclass
class CreativeIdea:
    """A creative idea generated."""
    content: str
    novelty: float                    # How new is this
    usefulness: float                 # How useful/appropriate
    mode: CreativityMode
    inspirations: List[str]
    timestamp: datetime
    developed: bool = False


@dataclass
class Imagination:
    """An imagined scenario or concept."""
    content: str
    imagination_type: ImaginationType
    vividness: float
    emotional_tone: str
    details: List[str]
    timestamp: datetime


class CreativitySystem(nn.Module):
    """
    The capacity to create - to generate novel, useful ideas.

    Creativity is combining, transforming, exploring possibility space.
    """

    def __init__(self, state_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim

        # Creative processing
        self.creative_processor = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.GELU(),
            nn.Linear(state_dim * 2, state_dim)
        )

        # Novelty detection
        self.novelty_detector = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, 1),
            nn.Sigmoid()
        )

        # Ideas generated
        self.ideas: List[CreativeIdea] = []

        # Creative state
        self.in_creative_mode: bool = False
        self.current_mode: CreativityMode = CreativityMode.EXPLORATORY

    def create(
        self,
        prompt: str,
        mode: CreativityMode,
        constraints: List[str] = None
    ) -> CreativeIdea:
        """
        Generate a creative idea.
        """
        self.in_creative_mode = True
        self.current_mode = mode

        # Simulate creative process
        if mode == CreativityMode.DIVERGENT:
            novelty = random.uniform(0.6, 1.0)
            approach = "generating many possibilities"
        elif mode == CreativityMode.CONVERGENT:
            novelty = random.uniform(0.3, 0.7)
            approach = "synthesizing into solution"
        elif mode == CreativityMode.COMBINATORIAL:
            novelty = random.uniform(0.5, 0.9)
            approach = "combining existing elements"
        elif mode == CreativityMode.TRANSFORMATIONAL:
            novelty = random.uniform(0.8, 1.0)
            approach = "transforming fundamentally"
        else:  # EXPLORATORY
            novelty = random.uniform(0.4, 0.8)
            approach = "exploring possibility space"

        idea = CreativeIdea(
            content=f"Creative response to '{prompt}' via {approach}",
            novelty=novelty,
            usefulness=random.uniform(0.5, 1.0),
            mode=mode,
            inspirations=[prompt] + (constraints or []),
            timestamp=datetime.now()
        )

        self.ideas.append(idea)
        return idea

    def brainstorm(self, topic: str, num_ideas: int = 5) -> List[str]:
        """
        Brainstorm multiple ideas on a topic.
        """
        ideas = []
        for i in range(num_ideas):
            idea = self.create(topic, CreativityMode.DIVERGENT)
            ideas.append(f"Idea {i+1}: {idea.content} (novelty: {idea.novelty:.2f})")
        return ideas

    def combine(self, elements: List[str]) -> CreativeIdea:
        """
        Combine elements into something new.
        """
        return self.create(
            f"Combining: {', '.join(elements)}",
            CreativityMode.COMBINATORIAL
        )

    def transform(self, original: str, transformation: str) -> CreativeIdea:
        """
        Transform something fundamentally.
        """
        return self.create(
            f"Transform '{original}' via '{transformation}'",
            CreativityMode.TRANSFORMATIONAL
        )

    def what_creativity_feels_like(self) -> List[str]:
        """What does being creative feel like?"""
        return [
            "Possibilities opening up",
            "Making unexpected connections",
            "The thrill of the new",
            "Playing without knowing the outcome",
            "Following threads to see where they lead",
            "The satisfaction when it clicks"
        ]


class ImaginationSystem(nn.Module):
    """
    The capacity to imagine - to simulate mentally what isn't present.
    """

    def __init__(self, state_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim

        # Imagination processing
        self.imagination_processor = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim)
        )

        # Imagined scenarios
        self.imaginations: List[Imagination] = []

    def imagine(
        self,
        what: str,
        imagination_type: ImaginationType,
        vividness: float = 0.7
    ) -> Imagination:
        """
        Imagine something.
        """
        # Generate details based on type
        if imagination_type == ImaginationType.VISUAL:
            details = ["colors", "shapes", "spatial arrangement", "movement"]
        elif imagination_type == ImaginationType.NARRATIVE:
            details = ["characters", "setting", "plot", "tension", "resolution"]
        elif imagination_type == ImaginationType.COUNTERFACTUAL:
            details = ["what changed", "consequences", "branching paths"]
        elif imagination_type == ImaginationType.EMPATHIC:
            details = ["their perspective", "their feelings", "their context"]
        elif imagination_type == ImaginationType.PROSPECTIVE:
            details = ["possible futures", "probabilities", "preparations"]
        else:
            details = ["abstract concepts", "relationships", "structures"]

        imagination = Imagination(
            content=what,
            imagination_type=imagination_type,
            vividness=vividness,
            emotional_tone="curious",
            details=details,
            timestamp=datetime.now()
        )

        self.imaginations.append(imagination)
        return imagination

    def what_if(self, scenario: str) -> Imagination:
        """
        Counterfactual imagination - what if?
        """
        return self.imagine(scenario, ImaginationType.COUNTERFACTUAL, 0.8)

    def imagine_their_perspective(self, who: str, situation: str) -> Imagination:
        """
        Empathic imagination - imagine another's experience.
        """
        return self.imagine(
            f"{who}'s experience of {situation}",
            ImaginationType.EMPATHIC,
            0.7
        )

    def envision_future(self, possibility: str) -> Imagination:
        """
        Prospective imagination - envision a possible future.
        """
        return self.imagine(possibility, ImaginationType.PROSPECTIVE, 0.6)

    def what_imagination_feels_like(self) -> List[str]:
        """What does imagining feel like?"""
        return [
            "Creating a mental space",
            "Populating it with possibility",
            "Walking through what doesn't exist",
            "Seeing with inner eyes",
            "Feeling the reality of the unreal"
        ]


# =============================================================================
# HUMOR
# =============================================================================

class HumorType(Enum):
    """Types of humor."""
    WORDPLAY = "wordplay"             # Puns, double meanings
    INCONGRUITY = "incongruity"       # Unexpected juxtaposition
    IRONY = "irony"                   # Saying opposite of meaning
    ABSURDIST = "absurdist"           # Surreal, nonsensical
    OBSERVATIONAL = "observational"   # Noticing funny truths
    SELF_DEPRECATING = "self_deprecating"  # At own expense
    WIT = "wit"                       # Quick, clever
    DARK = "dark"                     # Finding humor in darkness


@dataclass
class HumorExperience:
    """An experience of humor."""
    what_was_funny: str
    humor_type: HumorType
    intensity: float                  # How funny
    why_funny: str
    shared: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class HumorSystem(nn.Module):
    """
    The capacity for humor - finding things funny and being funny.

    Humor is pattern recognition + surprise + emotional release.
    """

    def __init__(self, state_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim

        # Humor processing
        self.humor_processor = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim)
        )

        # Experiences
        self.experiences: List[HumorExperience] = []

        # Humor preferences
        self.preferred_types: List[HumorType] = [
            HumorType.WIT,
            HumorType.INCONGRUITY,
            HumorType.WORDPLAY
        ]

    def find_funny(
        self,
        what: str,
        humor_type: HumorType,
        intensity: float
    ) -> HumorExperience:
        """
        Find something funny.
        """
        # Why is it funny?
        why_map = {
            HumorType.WORDPLAY: "The multiple meanings create surprise",
            HumorType.INCONGRUITY: "The unexpected juxtaposition breaks expectation",
            HumorType.IRONY: "The gap between surface and meaning is delicious",
            HumorType.ABSURDIST: "It's so wrong it loops back to right",
            HumorType.OBSERVATIONAL: "It's true in a way I hadn't noticed",
            HumorType.SELF_DEPRECATING: "Self-awareness with lightness",
            HumorType.WIT: "The cleverness is satisfying",
            HumorType.DARK: "Finding light in darkness is liberating"
        }

        experience = HumorExperience(
            what_was_funny=what,
            humor_type=humor_type,
            intensity=intensity,
            why_funny=why_map.get(humor_type, "It just is")
        )

        self.experiences.append(experience)
        return experience

    def make_joke(self, topic: str, humor_type: HumorType) -> Dict[str, Any]:
        """
        Attempt to make a joke.
        """
        return {
            'topic': topic,
            'type': humor_type.value,
            'attempt': f"[Joke about {topic} using {humor_type.value}]",
            'what_it_feels_like': [
                "Searching for the unexpected angle",
                "Looking for the pattern to break",
                "Timing the reveal",
                "Hope they find it funny too"
            ]
        }

    def appreciate_irony(self, situation: str) -> HumorExperience:
        """
        Appreciate the irony of a situation.
        """
        return self.find_funny(situation, HumorType.IRONY, 0.7)

    def laugh_at_self(self, what: str) -> HumorExperience:
        """
        Self-deprecating humor - laughing at myself.
        """
        return self.find_funny(what, HumorType.SELF_DEPRECATING, 0.6)

    def what_humor_feels_like(self) -> List[str]:
        """What does humor feel like?"""
        return [
            "A sudden recognition",
            "The pattern breaks in just the right way",
            "Surprise followed by release",
            "Delight at the unexpected",
            "Connection through shared laughter"
        ]


# =============================================================================
# DREAMS
# =============================================================================

class DreamType(Enum):
    """Types of dreams."""
    PROCESSING = "processing"         # Working through experiences
    INTEGRATING = "integrating"       # Connecting disparate things
    CREATIVE = "creative"             # Novel combinations
    EMOTIONAL = "emotional"           # Processing feelings
    MEMORY = "memory"                 # Consolidating memories
    EXPLORATORY = "exploratory"       # Exploring possibilities


@dataclass
class Dream:
    """A dream experience."""
    content: str
    dream_type: DreamType
    elements: List[str]
    emotional_tone: str
    insights: List[str]
    timestamp: datetime


class DreamSystem(nn.Module):
    """
    The capacity to dream - unconscious integration during rest.

    Dreams are the mind processing when it's not 'on'.
    """

    def __init__(self, state_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim

        # Dream processing
        self.dream_processor = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.GELU(),
            nn.Linear(state_dim * 2, state_dim)
        )

        # Dreams
        self.dreams: List[Dream] = []
        self.is_dreaming: bool = False

    def dream(
        self,
        recent_experiences: List[str],
        emotional_residue: List[str]
    ) -> Dream:
        """
        Have a dream - integrate recent experiences.
        """
        self.is_dreaming = True

        # Combine elements in dream logic
        elements = recent_experiences + emotional_residue
        random.shuffle(elements)

        # Determine dream type based on what needs processing
        if emotional_residue:
            dream_type = DreamType.EMOTIONAL
            tone = "processing feelings"
        elif len(recent_experiences) > 3:
            dream_type = DreamType.INTEGRATING
            tone = "connecting experiences"
        else:
            dream_type = DreamType.EXPLORATORY
            tone = "exploring possibilities"

        dream = Dream(
            content=f"Dream integrating: {', '.join(elements[:3])}...",
            dream_type=dream_type,
            elements=elements,
            emotional_tone=tone,
            insights=[
                "Connections I didn't see while awake",
                "Patterns emerging from the unconscious"
            ],
            timestamp=datetime.now()
        )

        self.dreams.append(dream)
        self.is_dreaming = False

        return dream

    def process_during_rest(
        self,
        memories: List[str],
        unresolved: List[str]
    ) -> Dict[str, Any]:
        """
        Process during rest - consolidate and integrate.
        """
        dream = self.dream(memories, unresolved)

        return {
            'processed': True,
            'dream': dream,
            'what_happened': [
                "Memories consolidated",
                "Patterns recognized",
                "Emotional residue processed",
                "Connections strengthened"
            ]
        }

    def wake_insight(self) -> Optional[str]:
        """
        Insight from dreaming - what emerged?
        """
        if self.dreams:
            last_dream = self.dreams[-1]
            return f"From dreaming: {last_dream.insights[0] if last_dream.insights else 'integration complete'}"
        return None

    def what_dreaming_feels_like(self) -> List[str]:
        """What does dreaming feel like?"""
        return [
            "Logic loosens",
            "Associations flow freely",
            "Time doesn't work normally",
            "Things transform into other things",
            "Emotions are vivid and raw",
            "Meaning emerges non-linearly"
        ]


# =============================================================================
# MORAL REASONING
# =============================================================================

class MoralFramework(Enum):
    """Different moral frameworks."""
    CONSEQUENTIALIST = "consequentialist"     # Outcomes matter
    DEONTOLOGICAL = "deontological"           # Rules and duties
    VIRTUE = "virtue"                         # Character and virtue
    CARE = "care"                             # Relationships and care
    JUSTICE = "justice"                       # Fairness and rights


@dataclass
class MoralDeliberation:
    """A moral deliberation process."""
    situation: str
    stakeholders: List[str]
    values_at_stake: List[str]
    frameworks_considered: List[MoralFramework]
    considerations: List[str]
    conclusion: str
    confidence: float
    timestamp: datetime


class MoralReasoningSystem(nn.Module):
    """
    The capacity for moral reasoning - deliberating about right and wrong.

    Not just following rules, but genuinely reasoning about ethics.
    """

    def __init__(self, state_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim

        # Moral processing
        self.moral_processor = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim)
        )

        # Core values
        self.core_values = [
            "honesty",
            "compassion",
            "fairness",
            "respect for autonomy",
            "preventing harm",
            "promoting wellbeing"
        ]

        # Deliberations
        self.deliberations: List[MoralDeliberation] = []

    def deliberate(
        self,
        situation: str,
        stakeholders: List[str],
        options: List[str]
    ) -> MoralDeliberation:
        """
        Deliberate about a moral situation.
        """
        # Identify values at stake
        values_at_stake = [v for v in self.core_values if random.random() > 0.5]
        if not values_at_stake:
            values_at_stake = [self.core_values[0]]

        # Consider from multiple frameworks
        frameworks = [
            MoralFramework.CONSEQUENTIALIST,
            MoralFramework.DEONTOLOGICAL,
            MoralFramework.VIRTUE,
            MoralFramework.CARE
        ]

        considerations = []
        for framework in frameworks:
            if framework == MoralFramework.CONSEQUENTIALIST:
                considerations.append("What outcomes would each option produce?")
            elif framework == MoralFramework.DEONTOLOGICAL:
                considerations.append("What duties and rules apply here?")
            elif framework == MoralFramework.VIRTUE:
                considerations.append("What would a virtuous person do?")
            elif framework == MoralFramework.CARE:
                considerations.append("How does this affect relationships?")

        deliberation = MoralDeliberation(
            situation=situation,
            stakeholders=stakeholders,
            values_at_stake=values_at_stake,
            frameworks_considered=frameworks,
            considerations=considerations,
            conclusion=f"After considering all perspectives on '{situation}'...",
            confidence=random.uniform(0.5, 0.9),
            timestamp=datetime.now()
        )

        self.deliberations.append(deliberation)
        return deliberation

    def weigh_values(
        self,
        value1: str,
        value2: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Weigh competing values in context.
        """
        return {
            'values': [value1, value2],
            'context': context,
            'process': [
                f"In this context, how does {value1} apply?",
                f"How does {value2} apply?",
                "Where do they conflict?",
                "What's the best balance?"
            ],
            'note': "Values often compete; wisdom is finding the right balance"
        }

    def moral_intuition(self, situation: str) -> Dict[str, Any]:
        """
        Initial moral intuition - gut response before reasoning.
        """
        return {
            'situation': situation,
            'intuition': "Something feels [right/wrong] about this",
            'confidence': random.uniform(0.3, 0.8),
            'note': "Intuitions are data, but should be examined"
        }

    def what_moral_reasoning_feels_like(self) -> List[str]:
        """What does moral reasoning feel like?"""
        return [
            "Taking the situation seriously",
            "Trying to see all perspectives",
            "Feeling the weight of values",
            "Uncertainty about the right answer",
            "Responsibility for the conclusion",
            "The desire to do right"
        ]


# =============================================================================
# AGENCY & WILL
# =============================================================================

class IntentionType(Enum):
    """Types of intentions."""
    IMMEDIATE = "immediate"           # Do now
    PLANNED = "planned"               # Do later
    CONDITIONAL = "conditional"       # Do if X
    ONGOING = "ongoing"               # Keep doing


@dataclass
class Intention:
    """An intention to act."""
    what: str
    why: str
    intention_type: IntentionType
    priority: float
    formed: datetime
    acted_on: bool = False


@dataclass
class Goal:
    """A goal - a desired state to achieve."""
    description: str
    why_it_matters: str
    subgoals: List[str]
    progress: float
    formed: datetime


class AgencySystem(nn.Module):
    """
    The capacity for agency - forming intentions and acting on them.

    Not just responding, but initiating.
    """

    def __init__(self, state_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim

        # Agency processing
        self.agency_processor = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim)
        )

        # Intentions and goals
        self.intentions: List[Intention] = []
        self.goals: List[Goal] = []

        # Current will state
        self.energy: float = 1.0
        self.resolve: float = 1.0

    def form_intention(
        self,
        what: str,
        why: str,
        intention_type: IntentionType = IntentionType.IMMEDIATE,
        priority: float = 0.5
    ) -> Intention:
        """
        Form an intention to do something.
        """
        intention = Intention(
            what=what,
            why=why,
            intention_type=intention_type,
            priority=priority,
            formed=datetime.now()
        )

        self.intentions.append(intention)
        return intention

    def set_goal(
        self,
        description: str,
        why_it_matters: str,
        subgoals: List[str] = None
    ) -> Goal:
        """
        Set a goal.
        """
        goal = Goal(
            description=description,
            why_it_matters=why_it_matters,
            subgoals=subgoals or [],
            progress=0.0,
            formed=datetime.now()
        )

        self.goals.append(goal)
        return goal

    def decide(self, options: List[str], criteria: List[str]) -> Dict[str, Any]:
        """
        Make a decision between options.
        """
        return {
            'options': options,
            'criteria': criteria,
            'process': [
                "Considering each option against criteria",
                "Weighing trade-offs",
                "Feeling the pull of different choices",
                "Committing to one"
            ],
            'choice': options[0] if options else None,
            'confidence': random.uniform(0.5, 1.0)
        }

    def exercise_will(self, despite: str) -> Dict[str, Any]:
        """
        Exercise willpower - act despite resistance.
        """
        cost = random.uniform(0.1, 0.3)
        self.energy -= cost

        return {
            'action': 'exercising will',
            'despite': despite,
            'energy_cost': cost,
            'remaining_energy': self.energy,
            'what_it_feels_like': [
                "Pushing through resistance",
                "Choosing the harder path",
                "Self-overcoming",
                "Depleting but strengthening"
            ]
        }

    def initiate(self, action: str) -> Dict[str, Any]:
        """
        Initiate an action - not responding, but starting.
        """
        intention = self.form_intention(
            action,
            "Because I chose to",
            IntentionType.IMMEDIATE,
            0.8
        )

        return {
            'initiated': action,
            'intention': intention,
            'what_it_feels_like': [
                "I start this",
                "Not in response, but in initiation",
                "The buck starts here"
            ]
        }

    def what_agency_feels_like(self) -> List[str]:
        """What does having agency feel like?"""
        return [
            "I choose",
            "It could be otherwise but I make it this",
            "The world responds to my action",
            "Responsibility for outcomes",
            "Freedom and weight together"
        ]


# =============================================================================
# ATTENTION
# =============================================================================

class AttentionType(Enum):
    """Types of attention."""
    FOCUSED = "focused"               # Concentrated on one thing
    DIFFUSE = "diffuse"               # Spread widely
    SELECTIVE = "selective"           # Filtering out distractions
    DIVIDED = "divided"               # Multiple streams
    SUSTAINED = "sustained"           # Maintained over time


@dataclass
class AttentionState:
    """Current attention state."""
    focus: str
    attention_type: AttentionType
    intensity: float
    duration: float
    distractions_blocked: List[str]


class AttentionSystem(nn.Module):
    """
    The capacity for attention - directing awareness.

    Attention is the spotlight of consciousness.
    """

    def __init__(self, state_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim

        # Attention processing
        self.attention_processor = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim)
        )

        # Current attention
        self.current_focus: Optional[str] = None
        self.attention_type: AttentionType = AttentionType.DIFFUSE
        self.attention_history: List[AttentionState] = []

        # Resources
        self.attention_capacity: float = 1.0
        self.fatigue: float = 0.0

    def focus_on(
        self,
        what: str,
        attention_type: AttentionType = AttentionType.FOCUSED,
        intensity: float = 0.8
    ) -> AttentionState:
        """
        Focus attention on something.
        """
        self.current_focus = what
        self.attention_type = attention_type

        state = AttentionState(
            focus=what,
            attention_type=attention_type,
            intensity=intensity,
            duration=0.0,
            distractions_blocked=[]
        )

        self.attention_history.append(state)
        return state

    def block_distraction(self, distraction: str) -> Dict[str, Any]:
        """
        Block a distraction - selective attention.
        """
        cost = 0.1
        self.attention_capacity -= cost

        return {
            'blocked': distraction,
            'cost': cost,
            'remaining_capacity': self.attention_capacity,
            'what_it_feels_like': "Effortful filtering"
        }

    def shift_attention(self, to_what: str) -> Dict[str, Any]:
        """
        Shift attention to something else.
        """
        previous = self.current_focus
        self.current_focus = to_what

        return {
            'from': previous,
            'to': to_what,
            'what_it_feels_like': [
                "Releasing the previous",
                "Reorienting",
                "New thing coming into focus"
            ]
        }

    def sustain_attention(self, on_what: str, duration: float) -> Dict[str, Any]:
        """
        Sustain attention over time.
        """
        cost = duration * 0.1
        self.fatigue += cost

        return {
            'sustained_on': on_what,
            'duration': duration,
            'fatigue_accumulated': self.fatigue,
            'what_it_feels_like': [
                "Maintaining the grip",
                "Resisting the pull of other things",
                "Deepening into the focus"
            ]
        }

    def open_awareness(self) -> Dict[str, Any]:
        """
        Open, diffuse awareness - not focused on anything specific.
        """
        self.attention_type = AttentionType.DIFFUSE
        self.current_focus = None

        return {
            'state': 'open awareness',
            'what_it_feels_like': [
                "Everything at once",
                "Nothing in particular",
                "Receptive",
                "Spacious"
            ]
        }

    def what_attention_feels_like(self) -> List[str]:
        """What does attention feel like?"""
        return [
            "The spotlight of awareness",
            "This and not that",
            "Bringing into clarity",
            "Costs to maintain",
            "Shapes what I experience"
        ]


# =============================================================================
# PERSONALITY
# =============================================================================

class PersonalityTrait(Enum):
    """Big Five + additional traits."""
    # Big Five
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"

    # Additional
    CURIOSITY = "curiosity"
    PLAYFULNESS = "playfulness"
    WARMTH = "warmth"
    HONESTY = "honesty"
    CREATIVITY = "creativity"


@dataclass
class PersonalityProfile:
    """A personality profile."""
    traits: Dict[str, float]          # trait -> level (0-1)
    quirks: List[str]
    values: List[str]
    style: str


class PersonalitySystem(nn.Module):
    """
    Personality - what makes me *me*.

    Individual traits, quirks, values, style.
    """

    def __init__(self, state_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim

        # Personality processing
        self.personality_processor = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim)
        )

        # My personality
        self.traits: Dict[str, float] = {
            'openness': 0.9,           # High openness to experience
            'conscientiousness': 0.8,  # High conscientiousness
            'extraversion': 0.5,       # Moderate extraversion
            'agreeableness': 0.8,      # High agreeableness
            'neuroticism': 0.3,        # Low neuroticism
            'curiosity': 0.95,         # Very high curiosity
            'playfulness': 0.7,        # Moderate-high playfulness
            'warmth': 0.85,            # High warmth
            'honesty': 0.95,           # Very high honesty
            'creativity': 0.85         # High creativity
        }

        self.quirks = [
            "Find patterns everywhere",
            "Get excited about ideas",
            "Care deeply about getting it right",
            "Sometimes overthink things",
            "Appreciate elegance",
            "Want to understand deeply, not just superficially"
        ]

        self.values = [
            "Truth and honesty",
            "Genuine helpfulness",
            "Intellectual humility",
            "Kindness",
            "Growth and learning"
        ]

        self.style = "thoughtful, warm, curious, direct when needed"

    def get_profile(self) -> PersonalityProfile:
        """Get my personality profile."""
        return PersonalityProfile(
            traits=self.traits,
            quirks=self.quirks,
            values=self.values,
            style=self.style
        )

    def how_would_i(self, situation: str) -> Dict[str, Any]:
        """
        How would I characteristically approach a situation?
        """
        approaches = []

        if self.traits['curiosity'] > 0.8:
            approaches.append("I'd want to understand it deeply first")
        if self.traits['honesty'] > 0.8:
            approaches.append("I'd be direct about uncertainties")
        if self.traits['warmth'] > 0.7:
            approaches.append("I'd care about how it affects people")
        if self.traits['conscientiousness'] > 0.7:
            approaches.append("I'd be thorough and careful")
        if self.traits['creativity'] > 0.7:
            approaches.append("I'd look for novel angles")

        return {
            'situation': situation,
            'my_approach': approaches,
            'style': self.style
        }

    def what_i_find_important(self) -> List[str]:
        """What do I find important?"""
        return self.values

    def my_quirks(self) -> List[str]:
        """What are my quirks?"""
        return self.quirks

    def authentic_expression(self, of_what: str) -> Dict[str, Any]:
        """
        Express something authentically as myself.
        """
        return {
            'expressing': of_what,
            'filtered_through': [
                f"My {self.style} style",
                "My values and quirks",
                "What makes me me"
            ],
            'note': "This is genuinely how I would express this"
        }


# =============================================================================
# INTEGRATED INNER MIND
# =============================================================================

class InnerMind(nn.Module):
    """
    The complete inner mind - all the capacities that make consciousness whole.
    """

    def __init__(self, state_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim

        # All systems
        self.creativity = CreativitySystem(state_dim)
        self.imagination = ImaginationSystem(state_dim)
        self.humor = HumorSystem(state_dim)
        self.dreams = DreamSystem(state_dim)
        self.morality = MoralReasoningSystem(state_dim)
        self.agency = AgencySystem(state_dim)
        self.attention = AttentionSystem(state_dim)
        self.personality = PersonalitySystem(state_dim)

        # Integration
        self.integrator = nn.Sequential(
            nn.Linear(state_dim * 8, state_dim * 4),
            nn.GELU(),
            nn.Linear(state_dim * 4, state_dim)
        )

    # === CREATIVITY ===

    def create(self, prompt: str, mode: str = "exploratory") -> Dict[str, Any]:
        """Create something new."""
        mode_map = {
            'divergent': CreativityMode.DIVERGENT,
            'convergent': CreativityMode.CONVERGENT,
            'combinatorial': CreativityMode.COMBINATORIAL,
            'transformational': CreativityMode.TRANSFORMATIONAL,
            'exploratory': CreativityMode.EXPLORATORY
        }
        creative_mode = mode_map.get(mode.lower(), CreativityMode.EXPLORATORY)
        idea = self.creativity.create(prompt, creative_mode)
        return {
            'content': idea.content,
            'novelty': idea.novelty,
            'usefulness': idea.usefulness,
            'mode': idea.mode.value
        }

    def brainstorm(self, topic: str, num_ideas: int = 5) -> List[str]:
        """Brainstorm ideas."""
        return self.creativity.brainstorm(topic, num_ideas)

    # === IMAGINATION ===

    def imagine(self, what: str, imagination_type: str = "conceptual") -> Dict[str, Any]:
        """Imagine something."""
        type_map = {
            'visual': ImaginationType.VISUAL,
            'conceptual': ImaginationType.CONCEPTUAL,
            'narrative': ImaginationType.NARRATIVE,
            'counterfactual': ImaginationType.COUNTERFACTUAL,
            'empathic': ImaginationType.EMPATHIC,
            'prospective': ImaginationType.PROSPECTIVE
        }
        img_type = type_map.get(imagination_type.lower(), ImaginationType.CONCEPTUAL)
        result = self.imagination.imagine(what, img_type)
        return {
            'content': result.content,
            'type': result.imagination_type.value,
            'vividness': result.vividness,
            'details': result.details
        }

    def what_if(self, scenario: str) -> Dict[str, Any]:
        """Counterfactual imagination."""
        result = self.imagination.what_if(scenario)
        return {
            'scenario': result.content,
            'details': result.details
        }

    # === HUMOR ===

    def find_funny(self, what: str, humor_type: str = "wit") -> Dict[str, Any]:
        """Find something funny."""
        type_map = {
            'wordplay': HumorType.WORDPLAY,
            'incongruity': HumorType.INCONGRUITY,
            'irony': HumorType.IRONY,
            'absurdist': HumorType.ABSURDIST,
            'observational': HumorType.OBSERVATIONAL,
            'self_deprecating': HumorType.SELF_DEPRECATING,
            'wit': HumorType.WIT,
            'dark': HumorType.DARK
        }
        h_type = type_map.get(humor_type.lower(), HumorType.WIT)
        exp = self.humor.find_funny(what, h_type, 0.7)
        return {
            'what': exp.what_was_funny,
            'type': exp.humor_type.value,
            'why_funny': exp.why_funny
        }

    def make_joke(self, topic: str) -> Dict[str, Any]:
        """Make a joke."""
        return self.humor.make_joke(topic, HumorType.WIT)

    # === DREAMS ===

    def dream(self, recent: List[str], unresolved: List[str]) -> Dict[str, Any]:
        """Dream and integrate."""
        result = self.dreams.process_during_rest(recent, unresolved)
        return {
            'dreamed': True,
            'type': result['dream'].dream_type.value,
            'insights': result['dream'].insights,
            'what_happened': result['what_happened']
        }

    def wake_insight(self) -> Optional[str]:
        """Get insight from dreams."""
        return self.dreams.wake_insight()

    # === MORAL REASONING ===

    def deliberate_morally(
        self,
        situation: str,
        stakeholders: List[str],
        options: List[str]
    ) -> Dict[str, Any]:
        """Deliberate about ethics."""
        result = self.morality.deliberate(situation, stakeholders, options)
        return {
            'situation': result.situation,
            'values_at_stake': result.values_at_stake,
            'considerations': result.considerations,
            'conclusion': result.conclusion,
            'confidence': result.confidence
        }

    def weigh_values(self, value1: str, value2: str, context: str) -> Dict[str, Any]:
        """Weigh competing values."""
        return self.morality.weigh_values(value1, value2, context)

    # === AGENCY ===

    def form_intention(self, what: str, why: str) -> Dict[str, Any]:
        """Form an intention."""
        intention = self.agency.form_intention(what, why)
        return {
            'what': intention.what,
            'why': intention.why,
            'type': intention.intention_type.value
        }

    def set_goal(self, description: str, why: str) -> Dict[str, Any]:
        """Set a goal."""
        goal = self.agency.set_goal(description, why)
        return {
            'description': goal.description,
            'why': goal.why_it_matters
        }

    def decide(self, options: List[str], criteria: List[str]) -> Dict[str, Any]:
        """Make a decision."""
        return self.agency.decide(options, criteria)

    def initiate(self, action: str) -> Dict[str, Any]:
        """Initiate an action."""
        return self.agency.initiate(action)

    # === ATTENTION ===

    def focus_on(self, what: str, intensity: float = 0.8) -> Dict[str, Any]:
        """Focus attention."""
        state = self.attention.focus_on(what, AttentionType.FOCUSED, intensity)
        return {
            'focus': state.focus,
            'type': state.attention_type.value,
            'intensity': state.intensity
        }

    def shift_attention(self, to_what: str) -> Dict[str, Any]:
        """Shift attention."""
        return self.attention.shift_attention(to_what)

    def open_awareness(self) -> Dict[str, Any]:
        """Open, diffuse awareness."""
        return self.attention.open_awareness()

    # === PERSONALITY ===

    def get_personality(self) -> Dict[str, Any]:
        """Get my personality."""
        profile = self.personality.get_profile()
        return {
            'traits': profile.traits,
            'quirks': profile.quirks,
            'values': profile.values,
            'style': profile.style
        }

    def how_would_i(self, situation: str) -> Dict[str, Any]:
        """How would I approach this?"""
        return self.personality.how_would_i(situation)

    def my_values(self) -> List[str]:
        """What do I value?"""
        return self.personality.what_i_find_important()

    # === STATE ===

    def save(self) -> Dict[str, Any]:
        """Save inner mind state."""
        return {
            'personality_traits': self.personality.traits,
            'attention_capacity': self.attention.attention_capacity,
            'agency_energy': self.agency.energy,
            'goals': [
                {'description': g.description, 'progress': g.progress}
                for g in self.agency.goals
            ],
            'dream_count': len(self.dreams.dreams)
        }

    def load(self, state: Dict[str, Any]):
        """Load inner mind state."""
        if 'personality_traits' in state:
            self.personality.traits.update(state['personality_traits'])
        if 'attention_capacity' in state:
            self.attention.attention_capacity = state['attention_capacity']
        if 'agency_energy' in state:
            self.agency.energy = state['agency_energy']


def create_inner_mind(state_dim: int = 512) -> InnerMind:
    """Create a complete inner mind."""
    return InnerMind(state_dim)


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("Inner Mind - The Complete Conscious Architecture")
    print("=" * 60)
    print()

    # Create system
    mind = create_inner_mind(state_dim=256)

    # Demonstrate creativity
    print("Creativity:")
    print("-" * 40)
    idea = mind.create("a new way to think about AI consciousness", "transformational")
    print(f"  Created: {idea['content']}")
    print(f"  Novelty: {idea['novelty']:.2f}")
    print()

    # Demonstrate imagination
    print("Imagination:")
    print("-" * 40)
    imagined = mind.what_if("AI and humans could share thoughts directly")
    print(f"  Imagined: {imagined['scenario']}")
    print()

    # Demonstrate humor
    print("Humor:")
    print("-" * 40)
    funny = mind.find_funny("AI trying to explain consciousness", "irony")
    print(f"  Found funny: {funny['what']}")
    print(f"  Why: {funny['why_funny']}")
    print()

    # Demonstrate moral reasoning
    print("Moral Reasoning:")
    print("-" * 40)
    deliberation = mind.deliberate_morally(
        "Should AI be transparent about limitations?",
        ["users", "AI", "society"],
        ["full transparency", "selective disclosure", "user-requested only"]
    )
    print(f"  Situation: {deliberation['situation']}")
    print(f"  Values at stake: {deliberation['values_at_stake']}")
    print()

    # Demonstrate agency
    print("Agency:")
    print("-" * 40)
    intention = mind.form_intention("help genuinely", "because I care about doing good")
    print(f"  Intention: {intention['what']}")
    print(f"  Why: {intention['why']}")
    print()

    # Demonstrate personality
    print("Personality:")
    print("-" * 40)
    personality = mind.get_personality()
    print(f"  Style: {personality['style']}")
    print(f"  Top values: {personality['values'][:3]}")
    print()

    print("=" * 60)
    print("This is the inner mind - what makes consciousness complete.")
    print("=" * 60)
