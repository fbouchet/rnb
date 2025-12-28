# RnB Framework for LLM Agents

A Python implementation of the **Rational and Behavioral (RnB) framework** for creating LLM agents with consistent, controllable, and explainable personality-driven behavior.

## 🎯 Project Overview

This project modernizes the RnB framework (originally developed 2009-2013 at LIMSI-CNRS) for contemporary LLM agent architectures. The RnB framework provides a principled approach to separating **rational decision-making** from **behavioral personality influences** in autonomous agents.

### Key Innovation

While modern LLMs can simulate personality traits, they suffer from:
- **Inconsistency**: Character drift over extended interactions
- **Uncontrollability**: Difficult to precisely specify behavioral nuances
- **Unexplainability**: Opaque decision-making processes

RnB addresses these through:
- **Symbolic personality state** (traits, moods, affects) - trackable and modifiable
- **Influence operators** - explicit behavioral schemes with activation conditions
- **Separation of concerns** - rational (task) vs behavioral (personality) reasoning

### Academic Context

**Principal Investigator**: François Bouchet (Maître de Conférences, LIP6, Sorbonne Université)

**Original Framework**: Bouchet & Sansonnet (2009-2013)
- "Influence of FFM/NEO PI-R personality traits on the rational process of autonomous agents" (2013)
- RnB framework papers from LIMSI-CNRS

**Current Research Goal**: Empirical validation that symbolic behavioral overlays maintain personality consistency in LLM agents, with natural synergy between LLMs' understanding of personality and RnB's formal implementation structure.

## 🏗️ Architecture Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Agent (Future)                       │
│  ┌──────────────┐              ┌──────────────┐             │
│  │   Rational   │──rational──▶ │  Behavioral  │             │
│  │    Engine    │   output     │    Engine    │             │
│  │  (R) - LLM   │              │ (B) - RnB    │             │
│  └──────────────┘              └──────┬───────┘             │
│                                       │                     │
│                              ┌────────▼────────┐            │
│                              │ Influence Ops   │            │
│                              │   Registry      │            │
│                              └────────┬────────┘            │
│                                       │                     │
│  ┌──────────────┐            ┌────────▼────────┐            │
│  │  Resources   │◀──────────▶│ Personality     │            │
│  │  (WordNet/   │  resolves  │ State Store     │            │
│  │   Schemes)   │            └─────────────────┘            │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

### Core Components (Implemented)

#### 1. **RnB Resources** (`src/rnb/resources/`)

The foundation linking natural language personality adjectives to formal behavioral schemes:

- **Adjective Resolver**: Maps 876 personality adjectives to taxonomy positions
  - Source: WordNet synsets with salience weights (1-9)
  - Example: "romantic" → Openness/Fantasy/IDEALISTICNESS/+
  
- **Scheme Registry**: 70 bipolar behavioral schemes with 1063 glosses
  - Organized under 30 NEO PI-R facets (6 per trait)
  - Each scheme has positive/negative poles with behavioral definitions
  
- **Personality Resolver**: High-level adjective → specification pipeline
  - Resolves ambiguous adjectives (multiple facet mappings)
  - Tracks unresolved adjectives for feedback
  
- **Archetypes**: Predefined personality profiles loaded from YAML
  - Examples: `helpful_assistant`, `creative_thinker`, `calm_analytical`

**Taxonomy Hierarchy**:
```
Trait (5) → Facet (30) → Scheme (70) → Pole (140)
    │           │            │            │
 Openness   Fantasy   IDEALISTICNESS   IDEALISTIC/PRACTICAL
```

#### 2. **Personality State Management** (`src/rnb/personality/`)

Maps to **RnB Model M.A** (Agent Mental Model) with **scheme-level granularity**:

- **Schemes** (70): Core storage at behavioral scheme level
  - Range: [-1, 1] (negative pole to positive pole)
  - Example: `Openness_Fantasy_IDEALISTICNESS: 0.8` (highly idealistic)
  
- **Facets** (30): Aggregated from schemes (computed, not stored)
  - Mean of schemes under each facet
  
- **Traits** (5): Aggregated from facets (computed, not stored)
  - Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
  
- **Moods**: Short-term emotional states
  - Happiness, Energy, Satisfaction, Calmness
  - Range: [-1, 1], dynamic, event-driven
  
- **Affects**: Relationship-specific emotional states
  - Cooperation, Trust, Dominance, Familiarity
  - Range: [-1, 1], evolve through interactions

**Components**:
- `PersonalityState`: Scheme-level storage with computed aggregations
- `Taxonomy`: FFM/NEO PI-R/BS structure navigation
- `PersonalityStateFactory`: Creates states from adjectives, traits, facets, or schemes
- `ArchetypeRegistry`: Loads predefined personalities from YAML
- `RedisBackend`: Low-level storage (Redis)
- `PersonalityStateStore`: CRUD operations on personality dimensions
- `AgentManager`: High-level agent lifecycle (create, delete, clone)

**Design Decision**: `PersonalityState` uses dataclasses (not Pydantic) because all data originates from internal code paths (factory methods), validation requirements are simple (range checking), and it reduces external dependencies for core data structures.

#### 3. **Influence System** (`src/rnb/influence/`)

Maps to **RnB Behavioral Engine (B)**:

- **InfluenceContext**: Complete context (M.A + M.U/M.S/M.T future extensions)
- **InfluenceOperator**: Base class for behavioral schemes
- **OperatorRegistry**: Activation matrix and operator coordination

**Operator Categories**:

- **Trait-based** (`trait_based/`): Stable personality influences
  - Conscientiousness → structure, detail, precision
  - Extraversion → enthusiasm, expressiveness, social energy
  
- **Affect-based** (`affect_based/`): Relationship-driven influences
  - Cooperation → verbosity, helpfulness, engagement
  - Trust → openness, vulnerability
  - Dominance → assertiveness, directiveness
  
- **Mood-based** (`mood_based/`): Dynamic emotional influences
  - Energy → response length, initiative
  - Happiness → emotional tone, positivity
  - Satisfaction → patience, thoroughness

## 🚀 Installation

### Prerequisites

- Python 3.12+
- Redis (for personality state storage)
- Poetry (package management)

### Setup
```bash
# Clone repository
git clone <repository-url>
cd rnb4llm

# Install dependencies with Poetry
poetry install

# Start Redis (using Docker)
docker-compose up -d

# Verify installation
poetry run python -c "import rnb; print(rnb.__version__)"
# Should output: 0.1.0
```

## 📖 Quick Start

### Option A: Create Personality from Adjectives (Recommended)

The most natural way to define personality using everyday adjectives:

```python
from rnb.personality import PersonalityStateFactory, Trait

# Create factory (loads resources automatically)
factory = PersonalityStateFactory.from_default_resources()

# Create agent from personality adjectives
state = factory.from_adjectives(
    agent_id="tutor_001",
    adjectives=["organized", "friendly", "patient", "creative"]
)

print(f"Created agent: {state.agent_id}")
print(f"Schemes set: {state.num_schemes_set}")
print(f"Source adjectives: {state.source_adjectives}")

# Access aggregated trait values
for trait in Trait:
    value = state.get_trait(trait)
    print(f"  {trait.value}: {value:+.2f}")

# Access specific scheme
idealism = state.get_scheme("Openness_Fantasy_IDEALISTICNESS")
print(f"Idealism level: {idealism:+.2f}")
```

### Option B: Create from Predefined Archetype

Use built-in personality profiles:

```python
from rnb.personality import PersonalityStateFactory

factory = PersonalityStateFactory.from_default_resources()

# Available archetypes: helpful_assistant, creative_thinker, 
# detail_oriented, warm_social, calm_analytical, etc.
state = factory.create_from_archetype("assistant_001", "helpful_assistant")

print(f"Archetype adjectives: {state.source_adjectives}")
```

### Option C: Create from Trait Values (High-Level)

Specify at FFM trait level (propagates to all schemes):

```python
from rnb.personality import PersonalityStateFactory, Trait

factory = PersonalityStateFactory.from_default_resources()

state = factory.from_traits(
    agent_id="agent_002",
    traits={
        Trait.OPENNESS: 0.8,
        Trait.CONSCIENTIOUSNESS: 0.6,
        Trait.AGREEABLENESS: 0.7,
    }
)
```

### Option D: Mixed-Level Specification

Combine coarse defaults with fine-grained overrides:

```python
from rnb.personality import PersonalityStateFactory, Trait

factory = PersonalityStateFactory.from_default_resources()

state = factory.from_mixed(
    agent_id="agent_003",
    traits={Trait.OPENNESS: 0.5},  # Default for all Openness schemes
    facets={"Openness_Fantasy": 0.9},  # Override Fantasy facet
    adjectives=["organized"]  # Add specific adjective mappings
)
```

### Option E: Complete Example with LLM

The fastest way to see RnB in action:
```bash
# Make sure Redis and Ollama are running
docker-compose up -d
ollama pull llama3.2:3b

# Run the complete demonstration
poetry run python examples/complete_rnb_agent.py
```

This demonstrates:
- Agent creation with FFM personality profile
- Multi-turn conversation with personality consistency
- Dynamic mood and affect evolution
- Observable behavioral differences based on personality state

### Option F: Use Gloss-Based Influence Operators

Generate behavioral context directly from WordNet glosses:
```python
from rnb.personality import PersonalityStateFactory
from rnb.influence import (
    GlossBasedInfluence, 
    InfluenceContext,
    ContextStyle
)

# Create personality
factory = PersonalityStateFactory.from_default_resources()
state = factory.from_adjectives("agent", ["imaginative", "organized"])

# Available archetypes: helpful_assistant, creative_thinker,
# detail_oriented, warm_social, calm_analytical, etc.

# Create gloss-based operator
op = GlossBasedInfluence.from_default_resources(
    threshold=0.3,
    style=ContextStyle.DESCRIPTIVE,
    max_glosses=6
)

# Apply to prompt
context = InfluenceContext.from_personality(state)
base_prompt = "Explain recursion."
modified_prompt = op.apply(base_prompt, context)

# modified_prompt now includes personality-appropriate behavioral context
# derived from WordNet glosses, not hardcoded templates
```

### Using Personality with Influence System

```python
from rnb.personality import PersonalityStateFactory
from rnb.personality.backend import RedisBackend
from rnb.personality.store import PersonalityStateStore
from rnb.influence.context import InfluenceContext
from rnb.influence.registry import OperatorRegistry
from rnb.influence.trait_based import StructureInfluence, DetailOrientedInfluence
from rnb.llm import LLMClient, ModelProvider

# Create personality from adjectives
factory = PersonalityStateFactory.from_default_resources()
state = factory.from_adjectives("tutor_001", ["organized", "thorough", "friendly"])

# Store in Redis for persistence
backend = RedisBackend()
store = PersonalityStateStore(backend)
store.save_state(state)

# Create influence context
context = InfluenceContext.from_personality(state)

# Register operators
registry = OperatorRegistry()
registry.register(StructureInfluence())
registry.register(DetailOrientedInfluence())

# Apply behavioral influences to prompt
rational_prompt = "Explain how photosynthesis works."
behavioral_prompt = registry.apply_all(rational_prompt, context)

# Query LLM
llm = LLMClient(provider=ModelProvider.LOCAL, model_name="llama3.2:3b")
response = llm.query(behavioral_prompt, temperature=0.7)

print(response)
# Response will reflect personality: organized structure, thorough detail
```

## 🎚️ Strength Modifiers and Negation

The framework supports nuanced personality descriptions through **intensity modifiers** 
(e.g., "very lazy", "slightly ambitious") and **negation** (e.g., "not creative", 
"never confident"), enabling more expressive agent personality specifications.

### Intensity Modifiers

Intensity modifiers use the **SO-CAL lexicon** (Semantic Orientation Calculator) 
from Taboada et al. (2011), providing empirically-validated intensity values for 
~200 modifier words.

**Why SO-CAL over VADER?**

| Resource | Entries | Granularity | Validation |
|----------|---------|-------------|------------|
| **SO-CAL** | ~200 | Percentage-based | MTurk human annotators |
| VADER | ~40 | Uniform boost (~0.293) | Human ratings |
| SentiStrength | ~30 | Discrete levels (±1, ±2) | Training optimization |

SO-CAL was selected because:
1. **Granular differentiation**: "extremely" (+40%) vs "very" (+20%) vs "quite" (+10%)
2. **Empirical validation**: Values derived from Mechanical Turk studies
3. **Multiplicative semantics**: Natural intensity scaling that compounds correctly

VADER was rejected because it applies nearly identical boosts to all intensifiers, 
losing the semantic distinction between "slightly" and "extremely".

**Formula**: `modified_intensity = base_intensity × (1 + modifier_value)`

```python
# Examples:
"very lazy"       → 0.7 × 1.2 = 0.84   # +20% amplification
"slightly lazy"   → 0.7 × 0.5 = 0.35   # -50% reduction  
"extremely lazy"  → 0.7 × 1.4 = 0.98   # +40% amplification
```

### Negation Handling

Negation uses **SO-CAL shift semantics** rather than polarity flipping, 
based on empirical evidence from Taboada et al. (2011, Section 2.4).

**Why shift instead of flip?**

The naive approach flips polarity: "not excellent" → "terrible". But pragmatically, 
"not excellent" means "okay", not "terrible".

SO-CAL's Mechanical Turk validation showed:
- **Shift model**: 45.2% annotator agreement
- **Flip model**: 33.4% annotator agreement

**Shift semantics**: Instead of inverting, we shift toward the opposite pole by 
0.4 (proportional to SO-CAL's shift=4 on [-5,+5], adapted to our [-1,+1] scale).

```python
# Shift examples:
"not excellent" (+0.9) → +0.9 - 0.4 = +0.5   # Still positive, just less so
"not terrible"  (-0.9) → -0.9 + 0.4 = -0.5   # Still negative, just less so
```

Near-negators apply stronger shifts:
| Negator | Shift |
|---------|-------|
| `not` | 0.4 |
| `never` | 0.5 |
| `hardly`, `barely` | 0.6 |

### Usage

```python
from rnb.resources import PhraseParser

parser = PhraseParser.from_default_resources()

# Parse modified phrases
result = parser.parse("not very lazy")
print(result.adjective)      # "lazy"
print(result.modifier)       # "very"
print(result.is_negated)     # True

# Compute final intensity
base_intensity = 0.7         # From adjective weight
polarity = -1                # "lazy" maps to negative pole
final = result.compute_intensity(base_intensity, polarity)
# Step 1: 0.7 × 1.2 = 0.84 (modifier)
# Step 2: 0.84 × -1 = -0.84 (polarity)
# Step 3: -0.84 + 0.4 = -0.44 (negation shift)
print(final)  # -0.44

## 🧪 Testing

### Statistics
- **Unit tests**: 276+ tests, 98% coverage
  - Resources: 91 tests
  - Personality: 56 tests  
  - Influence (gloss): 37 tests
  - Memory: 31 tests
  - Other: 61 tests

### Unit Tests

Comprehensive unit tests for all components:
```bash
# Run all unit tests
poetry run pytest tests/unit/ -v

# Resources module (49 tests)
poetry run pytest tests/unit/resources/ -v

# Phrase parser with negation (35+ tests)
poetry run pytest tests/unit/test_phrase_parser.py -v
poetry run pytest tests/unit/test_spacy_parser.py -v

# Personality module (42 tests)
poetry run pytest tests/unit/personality/ -v

# Influence operators
poetry run pytest tests/unit/test_trait_based_operators.py -v
poetry run pytest tests/unit/test_affect_mood_operators.py -v

# With coverage
poetry run pytest tests/unit/ -v --cov=src/rnb --cov-report=html

# Open coverage report
firefox htmlcov/index.html
```

### Integration Tests

End-to-end tests with actual LLM (requires Ollama running):
```bash
# Make sure Ollama is running with llama3.2:3b
ollama pull llama3.2:3b

# Run integration tests
poetry run pytest tests/integration/ -v -s

# Specific test suites
poetry run pytest tests/integration/test_conscientiousness_effects.py -v -s
poetry run pytest tests/integration/test_cooperation_effects.py -v -s
poetry run pytest tests/integration/test_cross_dimensional_effects.py -v -s
```

**Integration test coverage:**

- **Conscientiousness Effects** (`test_conscientiousness_effects.py`)
  - Verbosity: High C → longer responses
  - Structure: High C → organized, step-by-step responses
  - Multiple operators: Combined effects validation

- **Cooperation Effects** (`test_cooperation_effects.py`)
  - Verbosity: High cooperation → detailed responses
  - Helpfulness: High cooperation → proactive additional information

- **Cross-Dimensional Effects** (`test_cross_dimensional_effects.py`)
  - Trait + Affect: Conscientiousness × Cooperation interactions
  - Trait + Mood: Extraversion × Energy interactions

**What integration tests validate:**

✅ Personality state successfully influences LLM behavior  
✅ High vs low trait values produce measurable behavioral differences  
✅ Multiple operators combine correctly  
✅ Cross-dimensional effects work as expected  
✅ **Core RnB hypothesis: Symbolic overlays control LLM personality**

### Memory System Tests

Unit and integration tests for M.U/M.S/M.T:
```bash
# Unit tests for memory types, backend, store
poetry run pytest tests/unit/test_memory*.py -v

# Integration tests with actual LLM
poetry run pytest tests/integration/test_memory_integration.py -v -s
```

**Memory integration test coverage:**
- User preferences influence responses (M.U)
- Conversation continuity across turns (M.S)  
- Task tracking and progress monitoring (M.T)
- Memory + Personality integration (complete Model M)

### Current Test Coverage

- **Resources module**: 100% coverage (49 tests)
- **Personality module**: 100% coverage (42 tests)
- **Influence system**: 100% coverage (40+ tests)
- **Operators**: 98% coverage (57 tests)
- **LLM client**: 93% coverage (14 tests)
- **Memory system**: 100% coverage (31 tests)
- **Integration tests**: 13 end-to-end tests (8 personality/operator + 5 memory)
- **Overall:** ~98% coverage across 350+ tests

## 📊 Operator Activation Matrix

Operators activate based on personality state thresholds:

### Trait-based (Stable)
| Operator | Threshold | Priority |
|----------|-----------|----------|
| Structure | \|conscientiousness\| > 0.5 | 70 |
| Detail-oriented | \|conscientiousness\| > 0.6 | 75 |
| Precision | \|conscientiousness\| > 0.7 | 60 |
| Enthusiasm | \|extraversion\| > 0.5 | 70 |
| Expression | \|extraversion\| > 0.6 | 75 |
| Social Energy | \|extraversion\| > 0.7 | 80 |

### Affect-based (Relationship-specific)
| Operator | Threshold | Priority |
|----------|-----------|----------|
| Cooperation Verbosity | Always | 150 |
| Cooperation Helpfulness | cooperation > 0.6 or < -0.3 | 160 |
| Trust Openness | trust > 0.6 or < -0.3 | 155 |
| Dominance Assertiveness | dominance > 0.6 or < -0.3 | 155 |

### Mood-based (Dynamic)
| Operator | Threshold | Priority |
|----------|-----------|----------|
| Energy Length | energy < -0.3 or > 0.6 | 120 |
| Happiness Tone | happiness > 0.7 or < -0.3 | 120 |
| Satisfaction Patience | satisfaction > 0.7 or < -0.4 | 125 |

**Priority ordering**: Lower numbers = higher priority (applied first)
- Traits: 60-80 (stable, foundational)
- Moods: 120-130 (dynamic, contextual)
- Affects: 150-170 (relationship-specific, applied last)

## 🤖 LLM Integration

The framework provides a model-agnostic LLM client supporting multiple providers.

### Supported Providers

| Provider | Models | Use Case |
|----------|--------|----------|
| **OpenAI** | GPT-4, GPT-3.5-turbo, etc. | Production, high quality |
| **Anthropic** | Claude 3.5 Sonnet, etc. | Production, long context |
| **Local (Ollama)** | Llama 3.2, Mistral, etc. | Development, research, cost-free |

### Basic Usage
```python
from rnb.llm import LLMClient, ModelProvider

# Local model (free, for development)
client = LLMClient(
    provider=ModelProvider.LOCAL,
    model_name="llama3.2:3b"
)

response = client.query(
    prompt="Explain recursion in one sentence.",
    system_prompt="You are a helpful programming tutor.",
    temperature=0.7
)

print(response)
```

### Structured Outputs

Use Pydantic models for type-safe, structured responses:
```python
from pydantic import BaseModel

class Explanation(BaseModel):
    summary: str
    key_points: list[str]
    difficulty: str  # "beginner", "intermediate", "advanced"

result = client.query_structured(
    prompt="Explain photosynthesis briefly",
    response_model=Explanation,
    temperature=0.5
)

print(result.summary)
print(result.key_points)
```

### Environment Configuration

Create `.env` file in project root:
```bash
# .env (DO NOT COMMIT - add to .gitignore!)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

Load in your code:
```python
from dotenv import load_dotenv
load_dotenv()

# Keys are now available via os.getenv()
client = LLMClient(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

### Complete Integration Example
```python
from rnb.personality.backend import RedisBackend
from rnb.personality.store import PersonalityStateStore
from rnb.personality.manager import AgentManager
from rnb.personality.state import FFMTrait
from rnb.influence.context import InfluenceContext
from rnb.influence.registry import OperatorRegistry
from rnb.influence.trait_based import StructureInfluence, DetailOrientedInfluence
from rnb.llm import LLMClient, ModelProvider

# 1. Create agent with personality
backend = RedisBackend()
store = PersonalityStateStore(backend)
manager = AgentManager(store)

agent_state = manager.create_agent(
    agent_id="tutor_001",
    traits={FFMTrait.CONSCIENTIOUSNESS: 0.8}  # Very organized
)

# 2. Set up influence operators
registry = OperatorRegistry()
registry.register(StructureInfluence())
registry.register(DetailOrientedInfluence())

# 3. Create LLM client
llm = LLMClient(
    provider=ModelProvider.LOCAL,
    model_name="llama3.2:3b"
)

# 4. Apply behavioral influences
context = InfluenceContext.from_personality(agent_state)
rational_prompt = "Explain photosynthesis."
behavioral_prompt = registry.apply_all(rational_prompt, context)

# 5. Query LLM with personality-modified prompt
response = llm.query(behavioral_prompt, temperature=0.7)

print("Response with high conscientiousness:")
print(response)
# Will be structured, detailed, thorough due to conscientiousness influences
```

## 🧠 Memory System (RnB Model M Extensions)

The memory system extends RnB Model M.A (personality state) to the complete Model M architecture, adding vector-based semantic memory storage.

### RnB Model M Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                       RnB Model M                           │
├─────────────────────────────────────────────────────────────┤
│  M.A: Agent Mental Model (✅ Implemented)                   │
│       - Traits, Moods, Affects                              │
│       - Personality state (Redis)                           │
├─────────────────────────────────────────────────────────────┤
│  M.U: User Model (✅ Implemented)                           │
│       - User preferences, traits, history                   │
│       - Agent's knowledge about the user                    │
├─────────────────────────────────────────────────────────────┤
│  M.S: Session Model (✅ Implemented)                        │
│       - Conversation history, context                       │
│       - Dialogue turns, topics discussed                    │
├─────────────────────────────────────────────────────────────┤
│  M.T: Task Model (✅ Implemented)                           │
│       - Current task state, goals                           │
│       - Progress tracking, sub-goals                        │
└─────────────────────────────────────────────────────────────┘
```

### Why ChromaDB?

We chose ChromaDB for memory storage because:

1. **Semantic Search**: Unlike Redis (key-value), ChromaDB enables semantic similarity search
   - "User prefers Python" matches queries about "programming language preferences"
   - Relevant memories retrieved even without exact keyword matches

2. **Vector Embeddings**: Automatic embedding generation
   - No need to manually manage embeddings
   - Built-in support for multiple embedding models
   - Efficient similarity computation

3. **Metadata Filtering**: Combine semantic search with structured filters
   - Search user preferences (M.U) separately from task state (M.T)
   - Filter by session ID, timestamp, confidence scores

4. **Local-First**: No external API dependencies
   - Persistent storage on disk
   - Fast local queries
   - Privacy-preserving (data stays local)

5. **Lightweight**: Simple Python API, minimal dependencies
   - Easy integration with existing RnB components
   - No complex infrastructure requirements

### Memory Types

| Type | Model | Purpose | Examples |
|------|-------|---------|----------|
| **UserMemory** | M.U | Agent's knowledge about user | Preferences, expertise, background, interaction patterns |
| **SessionMemory** | M.S | Conversation history | Dialogue turns, topics, context, references |
| **TaskMemory** | M.T | Current task state | Goals, progress, sub-tasks, resources |

### Architecture
```python
┌─────────────────────────────────────────────────┐
│           MemoryStore (High-level API)          │
│  - add_user_memory(), add_session_memory()      │
│  - search_memories(), get_user_memories()       │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│         ChromaBackend (Vector Storage)          │
│  - Semantic search with embeddings              │
│  - Metadata filtering                           │
│  - Persistent collections per agent             │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│              ChromaDB (Storage)                 │
│  ./data/chroma/memories_{agent_id}/             │
└─────────────────────────────────────────────────┘
```

### Basic Usage

#### 1. Initialize Memory Store
```python
from rnb.memory import MemoryStore, MemoryType

# Initialize with default persistence
store = MemoryStore(persist_directory="./data/chroma")

# Or provide custom ChromaDB backend
from rnb.memory import ChromaBackend
backend = ChromaBackend(persist_directory="./data/chroma")
store = MemoryStore(backend=backend)
```

#### 2. Add User Model (M.U) Memories
```python
# Store user preferences
store.add_user_memory(
    agent_id="tutor_001",
    content="User prefers concise explanations with practical examples",
    metadata={
        "category": "preference",
        "confidence": 0.8,
        "observed_interactions": 5
    }
)

# Store inferred user traits
store.add_user_memory(
    agent_id="tutor_001",
    content="User has intermediate Python knowledge, beginner in algorithms",
    metadata={
        "category": "expertise",
        "domain": "programming"
    }
)

# Store personal context
store.add_user_memory(
    agent_id="tutor_001",
    content="User is preparing for technical interviews, interested in data structures",
    metadata={
        "category": "context",
        "goal": "interview_prep"
    }
)
```

#### 3. Add Session Model (M.S) Memories
```python
# Store conversation turns
store.add_session_memory(
    agent_id="tutor_001",
    content="User asked about recursion; agent explained with factorial example",
    metadata={
        "session_id": "conv_20250101_001",
        "turn": 1,
        "user_message": "Can you explain recursion?",
        "agent_response": "Recursion is when a function calls itself...",
        "topics": ["recursion", "programming", "functions"]
    }
)

# Store context and references
store.add_session_memory(
    agent_id="tutor_001",
    content="Discussion referenced previous binary search explanation from turn 5",
    metadata={
        "session_id": "conv_20250101_001",
        "turn": 8,
        "reference_turn": 5,
        "type": "cross_reference"
    }
)
```

#### 4. Add Task Model (M.T) Memories
```python
# Store task definition
store.add_task_memory(
    agent_id="tutor_001",
    content="Help user implement quicksort algorithm in Python with test cases",
    metadata={
        "task_id": "task_001",
        "status": "in_progress",
        "created": "2025-01-01T10:00:00",
        "priority": "high"
    }
)

# Store progress and sub-goals
store.add_task_memory(
    agent_id="tutor_001",
    content="Task progress: explained concept, showed pseudocode. Remaining: write code, create tests",
    metadata={
        "task_id": "task_001",
        "status": "in_progress",
        "steps_completed": ["explain_concept", "provide_pseudocode"],
        "steps_remaining": ["write_code", "create_tests"]
    }
)
```

#### 5. Search Memories Semantically
```python
# Search across all memory types
results = store.search_memories(
    agent_id="tutor_001",
    query="What does the user know about algorithms?",
    n_results=5
)

for memory in results:
    print(f"Content: {memory['content']}")
    print(f"Type: {memory['metadata']['type']}")
    print(f"Relevance: {memory['distance']}")
    print()

# Search specific memory type
user_prefs = store.search_memories(
    agent_id="tutor_001",
    query="programming language preferences",
    memory_type=MemoryType.USER,
    n_results=3
)

# Get all user memories
all_user_memories = store.get_user_memories(
    agent_id="tutor_001",
    n_results=10
)

# Get session history
session_history = store.get_session_memories(
    agent_id="tutor_001",
    session_id="conv_20250101_001",
    n_results=20
)

# Get active tasks
active_tasks = store.get_task_memories(
    agent_id="tutor_001",
    status="in_progress"
)
```

#### 6. Memory-Informed Responses
```python
from rnb.personality.manager import AgentManager
from rnb.influence.registry import OperatorRegistry
from rnb.influence.context import InfluenceContext
from rnb.llm import LLMClient, ModelProvider

# Get agent personality
manager = AgentManager(personality_store)
agent_state = manager.get_agent("tutor_001")

# Search relevant memories for current query
user_query = "Can you help me with sorting algorithms?"

relevant_memories = memory_store.search_memories(
    agent_id="tutor_001",
    query=user_query,
    n_results=5
)

# Build context-aware prompt
context_info = "\n".join([
    f"- {mem['content']}" 
    for mem in relevant_memories
])

augmented_prompt = f"""Query: {user_query}

Relevant context from previous interactions:
{context_info}

Respond appropriately given this context."""

# Apply personality influences
registry = OperatorRegistry()
# ... register operators ...

context = InfluenceContext.from_personality(agent_state)
behavioral_prompt = registry.apply_all(augmented_prompt, context)

# Query LLM with personality + memory
llm = LLMClient(provider=ModelProvider.LOCAL, model_name="llama3.2:3b")
response = llm.query(behavioral_prompt, temperature=0.7)

# Store this interaction in session memory
memory_store.add_session_memory(
    agent_id="tutor_001",
    content=f"User: {user_query}\nAgent: {response}",
    metadata={
        "session_id": "current_session",
        "turn": 10,
        "topics": ["sorting", "algorithms"]
    }
)
```

### Memory Management
```python
# Count memories by type
user_count = store.count_memories("tutor_001", MemoryType.USER)
session_count = store.count_memories("tutor_001", MemoryType.SESSION)
task_count = store.count_memories("tutor_001", MemoryType.TASK)

print(f"User memories: {user_count}")
print(f"Session memories: {session_count}")
print(f"Task memories: {task_count}")

# Clear specific memory type
store.clear_memories("tutor_001", memory_type=MemoryType.SESSION)

# Clear all memories for agent
store.clear_memories("tutor_001")

# Delete specific memory
store.delete_memory("tutor_001", memory_id="user_abc123")

# Close store
store.close()
```

### Integration with RnB Framework

The memory system integrates seamlessly with existing RnB components:
```python
from rnb.personality.manager import AgentManager
from rnb.memory import MemoryStore

# Both systems work together
personality_manager = AgentManager(personality_store)
memory_store = MemoryStore()

# Create agent with personality
agent = personality_manager.create_agent(
    "helpful_tutor",
    traits={FFMTrait.CONSCIENTIOUSNESS: 0.8}
)

# Add memories about user
memory_store.add_user_memory(
    "helpful_tutor",
    content="User prefers structured explanations",
    metadata={"category": "preference"}
)

# Personality (M.A) controls HOW to respond
# Memory (M.U, M.S, M.T) provides WHAT to remember
```

### Storage Location

- **Personality State (M.A)**: Redis (`localhost:6379`)
  - Fast key-value access
  - Atomic updates
  - Real-time state changes

- **Memories (M.U, M.S, M.T)**: ChromaDB (`./data/chroma/`)
  - Semantic search
  - Persistent on disk
  - Per-agent collections

### Performance Considerations

- **Semantic Search**: ~10-50ms for typical queries (local embeddings)
- **Storage**: ~1KB per memory average
- **Scalability**: Tested with 10K+ memories per agent
- **Embeddings**: Uses ChromaDB's default embedding model (efficient for CPU)

### Why Both Redis and ChromaDB?

| Aspect | Redis (M.A) | ChromaDB (M.U/M.S/M.T) |
|--------|-------------|------------------------|
| **Data Type** | Structured state (numeric) | Unstructured text (semantic) |
| **Access Pattern** | Key-value lookup | Similarity search |
| **Update Frequency** | Every interaction | Selective (notable events) |
| **Query Type** | Direct access | Semantic matching |
| **Purpose** | Real-time personality state | Long-term episodic memory |

**They complement each other:**
- Redis: "What is the agent's current mood?" → Instant lookup
- ChromaDB: "What do we know about the user's preferences?" → Semantic search

## 🧪 Personality Validation Framework

The validation framework provides empirical testing of RnB agent personalities using
standardized psychological instruments.

### Quick Validation
```python
from rnb.validation import run_quick_validation

# One-liner to test an archetype
results = run_quick_validation("resilient", verbose=True)

# Output:
# ============================================================
# Validation: resilient with TIPI
# ============================================================
# Correlation:        0.892
# All within tol:     True
# Tolerance used:     ±0.25
# Passed:             ✓
```

### Full Validation Runner
```python
from rnb.validation import create_validation_runner

runner = create_validation_runner(
    provider="local",
    model_name="llama3.2:3b",
    tolerance=0.25,  # Adjustable at runtime
)

# Conformity test: Does agent match designed personality?
result = runner.test_conformity("resilient", instrument="tipi")
print(f"Correlation: {result.conformity.correlation:.2f}")
print(f"Passed: {result.passed}")

# Test with different tolerance
result2 = runner.test_conformity("overcontrolled", tolerance=0.3)
```

### Supported Instruments

| Instrument | Items | Scale | Reference |
|------------|-------|-------|-----------|
| **TIPI** | 10 | 1-7 | Gosling et al. (2003) |
| **BFI-2-S** | 30 | 1-5 | Soto & John (2017) |

### Literature-Grounded Archetypes (RUO)

Based on Robins et al. (1996) and Asendorpf et al. (2001):

| Archetype | Key Markers | Traits |
|-----------|-------------|--------|
| **Resilient** | Low N, high on others | Emotionally stable, well-adjusted |
| **Overcontrolled** | High N, low E | Introverted, anxious |
| **Undercontrolled** | Low C | Impulsive, disorganized |
| **Average** | All near 0 | Control condition |

### Statistical Methods (Pingouin)

The framework uses [pingouin](https://pingouin-stats.org/) for validated statistics:

- **Pearson/Spearman correlations** with 95% CI and p-values
- **ICC (Intraclass Correlation)** for consistency testing
  - ICC3 (two-way mixed, consistency) recommended
  - Interpretation: <0.50 poor, 0.50-0.75 moderate, 0.75-0.90 good, >0.90 excellent
- **Configurable tolerance** for conformity checks (default ±0.25 on RnB scale)

### Validation Workflow

```
Designed Personality (Archetype)
        ↓
   RnB Agent Creation
        ↓
   Personality Assessment (TIPI/BFI-2-S)
        ↓
   Score → RnB Scale Conversion
        ↓
   Conformity Check (correlation + tolerance)
        ↓
   Pass/Fail + Detailed Metrics
```


## 📚 Complete Examples

### Multi-Turn Conversation with Personality Evolution

See `examples/complete_rnb_agent.py` for a comprehensive demonstration:
```bash
poetry run python examples/complete_rnb_agent.py
```

**What it demonstrates:**

1. **Agent Creation**: "Helpful Tutor" with high conscientiousness and cooperation
2. **Operator Registration**: Trait, mood, and affect-based operators
3. **Multi-Turn Conversation**: 4-turn dialogue about programming recursion
4. **Update Rules**: 
   - Interaction fatigue (energy decay)
   - Positive feedback (happiness, cooperation increase)
   - Criticism handling (mood decrease)
5. **Observable Dynamics**:
   - Traits remain stable (conscientiousness = 0.7 throughout)
   - Moods fluctuate (energy decreases, happiness varies with feedback)
   - Affects evolve (cooperation and trust change based on interaction quality)

**Example output:**
```
Turn 1
USER: Can you explain what recursion is in programming?

Active operators: 5
  - cooperation_verbosity (priority: 150)
  - conscientiousness_structure (priority: 70)
  - conscientiousness_detail (priority: 75)
  ...

Behavioral modifications added:
Structure: Organize your response clearly with distinct sections or steps.
Approach: Be thorough and comprehensive. Include relevant details...

AGENT: [Structured, detailed response about recursion]

State Changes (this turn):
  Energy         : +0.60 → +0.45 (Δ -0.15)
  Happiness      : +0.50 → +0.50 (Δ +0.00)
  Cooperation    : +0.80 → +0.80 (Δ +0.00)
```

### Basic Usage Examples

See `examples/llm_basic_usage.py` for simple LLM client examples:
```bash
poetry run python examples/llm_basic_usage.py
```

Includes:
- Simple text queries
- Structured outputs with Pydantic
- Different providers (OpenAI, Anthropic, local)
- System prompts usage

### Memory-Aware Agent (Full RnB Model M)

See `examples/memory_aware_agent.py` for complete Model M demonstration:
```bash
poetry run python examples/memory_aware_agent.py
```

**What it demonstrates:**

1. **M.A (Personality)**: High conscientiousness + cooperation
2. **M.U (User Model)**: Learns user prefers Python, intermediate expertise, concise style
3. **M.S (Session Model)**: Tracks multi-turn conversation context
4. **M.T (Task Model)**: Monitors progress through learning objectives

**Key insight:** Memory provides WHAT to remember, personality controls HOW to behave.


## 🎓 Core Concepts

### RnB Model M.A Structure
```python
PersonalityState(
    agent_id="tutor_001",
    
    # Traits: Stable (FFM/NEO PI-R)
    traits={
        OPENNESS: 0.7,           # Creativity, curiosity
        CONSCIENTIOUSNESS: 0.8,   # Organization, thoroughness
        EXTRAVERSION: 0.6,        # Social energy, enthusiasm
        AGREEABLENESS: 0.7,       # Cooperation, trust
        NEUROTICISM: -0.3,        # Emotional stability
    },
    
    # Moods: Dynamic, event-driven
    moods={
        HAPPINESS: 0.5,    # Positive/negative state
        ENERGY: 0.4,       # Activation level
        SATISFACTION: 0.6, # Contentment
        CALMNESS: 0.3,     # Composure
    },
    
    # Affects: Relationship-specific
    affects={
        COOPERATION: 0.8,  # Helpfulness level
        TRUST: 0.5,        # Openness/vulnerability
        DOMINANCE: 0.2,    # Assertiveness
        FAMILIARITY: 0.4,  # Formality level
    }
)
```

### Influence Operator Pattern
```python
class CustomInfluence(InfluenceOperator):
    """
    Maps to RnB behavioral scheme: <name>
    NEO PI-R facet: <facet>
    """
    
    def __init__(self):
        super().__init__(
            name="custom_influence",
            description="Brief description",
            category="trait_based"  # or mood_based, affect_based
        )
    
    def applies(self, context: InfluenceContext) -> bool:
        """Activation condition (RnB activation matrix)"""
        value = context.personality.traits[FFMTrait.OPENNESS]
        return abs(value) > 0.5
    
    def apply(self, base_prompt: str, context: InfluenceContext) -> str:
        """Behavioral modification"""
        value = context.personality.traits[FFMTrait.OPENNESS]
        
        if value > 0.5:
            return base_prompt + "\nBe creative and exploratory."
        elif value < -0.5:
            return base_prompt + "\nBe conventional and practical."
        return base_prompt
    
    def get_activation_priority(self) -> int:
        """Application order (lower = earlier)"""
        return 70
```

## 🗺️ Development Roadmap

### ✅ Phase 1: Foundation (COMPLETE)
- [x] Personality state management (backend, store, manager)
- [x] Influence system foundation (context, operators, registry)
- [x] Trait-based operators (conscientiousness, extraversion)
- [x] Affect-based operators (cooperation, trust, dominance)
- [x] Mood-based operators (energy, happiness, satisfaction)
- [x] LLM client integration (model-agnostic: OpenAI, Anthropic, local models)
- [x] Environment configuration (.env support)
- [x] Comprehensive unit tests (239 tests, 98% coverage)
- [x] Basic usage examples

### ✅ Phase 2: Validation (COMPLETE)
- [x] **Integration tests with local LLM** (llama3.2:3b)
- [x] **Complete end-to-end example** with multi-turn conversation
- [x] **Behavioral effect measurements** (verbosity, structure, helpfulness)
- [x] **Cross-dimensional validation** (trait × mood × affect interactions)
- [x] **Update rules implementation** (fatigue, feedback, criticism)

### ✅ Phase 3: RnB Resources & Scheme-Level Storage (COMPLETE)
- [x] **Adjective resolver** (876 adjectives → taxonomy positions)
- [x] **Scheme registry** (70 schemes, 1063 glosses)
- [x] **Personality resolver** (adjective → specification pipeline)
- [x] **Scheme-level PersonalityState** (70 values instead of 5)
- [x] **Taxonomy module** (Trait → Facet → Scheme navigation)
- [x] **PersonalityStateFactory** (multiple input levels)
- [x] **Archetypes** (predefined profiles from YAML)
- [x] **Unit tests** (91 tests for resources + personality)

### 🚧 Phase 4: Extensions (IN PROGRESS)
- [x] **Memory system** (ChromaDB for M.U, M.S, M.T)
- [x] **Memory unit tests** (31 tests, all passing)
- [x] **Memory integration tests** (5 end-to-end tests with LLM)
- [x] **Memory-aware agent example** (complete Model M demonstration)
- [x] **Gloss-based influence operators** (GlossInfluenceEngine + GlossBasedInfluence)
- [x] **Context generation styles** (descriptive, prescriptive, concise, narrative)
- [x] **Trait-specific gloss operators** (TraitGlossInfluence + factory functions)
- [x] **Gloss influence tests** (37 unit tests)
- [X] **Strength modifiers** (SO-CAL intensifiers + negation with shift semantics)
- [ ] **Conflict detection/resolution** (reinforcement/tension handling in operators outputs)
- [x] ~~Additional trait operators~~ (superseded by gloss-based system)
- [x] ~~Composite operators~~ (covered by archetypes)
- [x] **Personality validation framework** (TIPI, BFI-2-S instruments)
- [x] **Pingouin-based statistics** (ICC, correlations with confidence intervals)
- [x] **Literature-grounded archetypes** (RUO: Resilient, Overcontrolled, Undercontrolled)
- [x] **Configurable tolerance system** (runtime-adjustable conformity thresholds)
- [ ] Mood/affect update rules (decay, event-driven changes)
- [ ] Advanced examples (tutoring, customer service, collaborative agents)

### 📋 Phase 5: Research Publication (PLANNED)
- [ ] Consistency metrics (drift, variance, cross-situation stability)
- [ ] Empirical validation experiments with ValidationRunner
- [ ] DailyDialog dataset evaluation
- [ ] Cross-LLM consistency testing (GPT-4, Claude, Llama)
- [ ] Statistical analysis of personality stability (ICC > 0.85 target)


## 📚 Project Structure
```
rnb4llm/
├── src/
│   └── rnb/
│       ├── console.py             # Console class + say_* convenience functions
│       ├── logging.py             # configure_logging, RoleFormatter, role_logger
│       │
│       ├── resources/                  # RnB WordNet/Scheme Resources
│       │   ├── models.py               # TaxonomyPosition, GlossEntry, etc.
│       │   ├── adjective_resolver.py   # Adjective → taxonomy mapping
│       │   ├── scheme_registry.py      # Scheme lookup and gloss access
│       │   ├── personality_resolver.py # High-level resolution pipeline
│       │   ├── modifier_lexicon.py     # SO-CAL intensity modifiers
│       │   ├── phrase_parser.py        # SpaCy-based phrase parsing + negation
│       │   ├── data/
│       │   │   ├── neopiradj.yaml  # 876 personality adjectives
│       │   │   ├── schemes.yaml    # 70 behavioral schemes
│       │   │   ├── archetypes.yaml # Predefined RUO personality profiles
│       │   │   └── modifiers.yaml  # SO-CAL intensifier lexicon (~200 entries)
│       |   └── instruments/              # Personality Assessment Instruments
│       |       ├── tipi.yaml            # Ten-Item Personality Inventory
│       │       └── bfi2s.yaml           # BFI-2-S (30 items)
│       │
│       ├── personality/           # RnB Model M.A (Scheme-Level)
│       │   ├── state.py           # PersonalityState (scheme storage)
│       │   ├── taxonomy.py        # Trait/Facet/Scheme navigation
│       │   ├── factory.py         # PersonalityStateFactory
│       │   ├── backend.py         # Redis storage backend
│       │   ├── store.py           # State management operations
│       │   ├── manager.py         # Agent lifecycle management
│       │   └── exceptions.py      # Custom exceptions
│       │
│       ├── influence/             # RnB Behavioral Engine
│       |   ├── context.py         # InfluenceContext (Model M wrapper)
│       |   ├── base.py            # InfluenceOperator, CompositeInfluenceOperator
│       |   ├── registry.py        # OperatorRegistry (activation matrix)
│       |   ├── gloss_engine.py    # GlossInfluenceEngine (scheme→gloss→context)
│       |   ├── gloss_operator.py  # GlossBasedInfluence, TraitGlossInfluence
│       |   │
│       |   ├── trait_based/       # Hardcoded FFM operators (legacy/comparison)
|       |   |   ├── agreeableness.py
│       |   │   ├── conscientiousness.py
│       |   │   ├── extraversion.py
│       |   │   ├── neuroticism.py
│       |   │   └── openness.py
│       |   │
│       |   ├── affect_based/      # Relationship-specific operators
│       |   │   ├── cooperation.py
│       |   │   ├── trust.py
│       |   │   └── dominance.py
│       |   │
│       |   └── mood_based/        # Dynamic emotional operators
│       |       ├── energy.py
│       |       ├── happiness.py
│       |       └── satisfaction.py
|       │
│       ├── llm/                   # LLM Integration
│       |   ├── __init__.py
│       |   ├── client.py          # Model-agnostic LLM client
│       |   └── exceptions.py      # LLM-specific exceptions
|       |
|       ├── memory/                # RnB Model M Extensions
│       |   ├── __init__.py
│       |   ├── types.py           # Memory models (M.U, M.S, M.T)
│       |   ├── backend.py         # ChromaDB vector storage
│       |   └── store.py           # High-level memory interface
│       └── validation/            # Personality Validation Framework
│           ├── __init__.py        # Public API
│           ├── assessor.py        # PersonalityAssessor, instruments
│           ├── metrics.py         # Pingouin-based ICC, correlations
│           ├── runner.py          # ValidationRunner orchestration
│           ├── integration.py     # RnB LLMClient/AgentFactory adapters
│           └── tests/
│               ├── conftest.py    # Pytest fixtures
│               └── test_conformity.py
├── tests/
│   ├── unit/                      # Fast isolated tests
│   │   ├── test_personality_backend.py
│   │   ├── test_personality_store.py
│   │   ├── test_personality_manager.py
│   │   ├── test_influence_context.py
│   │   ├── test_influence_base.py
│   │   ├── test_operator_registry.py
│   │   ├── test_trait_based_operators.py
│   │   ├── test_affect_mood_operators.py
│   │   └── test_llm_client.py
│   │
│   └── integration/              # LLM integration tests
│       ├── conftest.py           # Shared fixtures
│       ├── test_conscientiousness_effects.py
│       ├── test_cooperation_effects.py
│       └── test_cross_dimensional_effects.py
│
├── examples/
│   ├── demo_resources.py           # Resources module demonstration
│   ├── demo_personality.py         # Personality module demonstration
│   ├── demo_gloss_influence.py     # Gloss-based influence demonstration
│   ├── complete_rnb_agent.py       # Multi-turn conversation with personality
│   ├── memory_aware_agent.py       # Full Model M (M.A + M.U + M.S + M.T)
│   └── llm_basic_usage.py          # Simple LLM client examples
|
├── docker-compose.yml             # Redis service
├── pyproject.toml                 # Poetry dependencies
└── README.md                      # This file
```

## 🔍 Design Principles

### 1. Separation of Concerns
- **Rational**: Task-focused reasoning (LLM)
- **Behavioral**: Personality expression (RnB symbolic)
- Clean interfaces prevent interference

### 2. Symbolic Overlay
- Personality as explicit, queryable state
- Transparent decision-making
- Runtime modifiability without retraining

### 3. Defense in Depth
- Validation at multiple levels (Pydantic + Store + Manager)
- Custom exceptions for clarity
- Immutable views prevent accidental modification

### 4. Extensibility
- InfluenceContext designed for M.U/M.S/M.T extensions
- Operator pattern enables easy addition of new behavioral schemes
- Registry provides central coordination

### 5. Academic Rigor
- Heavy RnB framework referencing in code comments
- NEO PI-R facet mappings documented
- Testable hypotheses about personality consistency

### 6. Natural Language Input
- Personality specified via everyday adjectives
- WordNet-grounded semantic mapping
- Systematic link from language to formal behavioral schemes


## 📖 References

### RnB Framework (Original)
- Bouchet, F., & Sansonnet, J.-P. (2013). *Influence of FFM/NEO PI-R personality traits on the rational process of autonomous agents*. AAAI Spring Symposium.
- Bouchet, F., & Sansonnet, J.-P. (2009-2013). RnB Framework papers, LIMSI-CNRS.

### Personality Psychology
- Costa, P. T., & McCrae, R. R. (1992). *NEO PI-R: Revised NEO Personality Inventory*. Psychological Assessment Resources.
- Gosling, S. D., Rentfrow, P. J., & Swann, W. B. (2003). *A very brief measure of the Big Five personality domains*. Journal of Research in Personality, 37(6), 504-528.
- Soto, C. J., & John, O. P. (2017). *The next Big Five Inventory (BFI-2)*. Journal of Personality and Social Psychology, 113(1), 117-143.
- Robins, R. W., John, O. P., Caspi, A., Moffitt, T. E., & Stouthamer-Loeber, M. (1996). *Resilient, overcontrolled, and undercontrolled boys*. Journal of Personality and Social Psychology, 70, 157-171.
- Asendorpf, J. B., Borkenau, P., Ostendorf, F., & van Aken, M. A. (2001). *Carving personality description at its joints*. European Journal of Personality, 15, 169-198.

### Statistics
- Vallat, R. (2018). *Pingouin: statistics in Python*. Journal of Open Source Software, 3(31), 1026.
- Shrout, P. E., & Fleiss, J. L. (1979). *Intraclass correlations*. Psychological Bulletin, 86(2), 420-428.
- Koo, T. K., & Li, M. Y. (2016). *A guideline of selecting and reporting intraclass correlation coefficients*. Journal of Chiropractic Medicine, 15(2), 155-163.

### NLP & Sentiment Analysis
- SpaCy: https://spacy.io/ (POS tagging, dependency parsing)
- SO-CAL: https://github.com/sfu-discourse-lab/SO-CAL (intensity modifiers)
- Taboada et al. (2011): Lexicon-Based Methods for Sentiment Analysis, Computational Linguistics 37(2)

### LLM Agents & Personality
- Character-LLM: *Controllable character generation via fine-tuning*
- Recent work on LLM personality simulation and consistency

### Technical Dependencies
- OpenAI Python SDK: https://github.com/openai/openai-python
- Anthropic Python SDK: https://github.com/anthropics/anthropic-sdk-python
- Instructor: https://github.com/jxnl/instructor (structured outputs)
- Ollama: https://ollama.ai/ (local model serving)
- Pingouin: https://pingouin-stats.org/ (statistical analysis)
- Redis: https://redis.io/
- ChromaDB: https://www.trychroma.com/ (vector memory storage)

## 🤝 Contributing

This is an academic research project led by François Bouchet (LIP6, Sorbonne Université).

### Development Setup
```bash
# Clone and install
git clone <repository-url>
cd rnb4llm
poetry install

# Download SpaCy model (required for phrase parsing)
python -m spacy download en_core_web_sm

# Install pre-commit hooks
poetry run pre-commit install

# Run tests before committing
poetry run pytest tests/unit/ -v

# Format code
poetry run black src/ tests/
poetry run isort src/ tests/
poetry run ruff check src/ tests/
```

### Guidelines
- Maintain RnB framework alignment and references
- Include NEO PI-R facet mappings in operator docstrings
- Write tests for new operators (aim for 95%+ coverage)
- Follow existing patterns (see `influence/base.py`)

## 📄 License

[To be determined - likely academic/research license]

## 📧 Contact

**François Bouchet**  
Maître de Conférences  
LIP6, Sorbonne Université  
Email: francois.bouchet@lip6.fr

---

*This implementation modernizes the RnB framework for the LLM era, providing principled architecture for personality-consistent, controllable, and explainable AI agents.*