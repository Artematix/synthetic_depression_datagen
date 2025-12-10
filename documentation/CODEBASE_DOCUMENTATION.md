# Synthetic Depression Screening Dialogue Generator - Complete Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Key Components](#key-components)
4. [Data Structures](#data-structures)
5. [Prompt Engineering](#prompt-engineering)
6. [Workflow](#workflow)
7. [Output Structure](#output-structure)
8. [How to Extend](#how-to-extend)
9. [Configuration](#configuration)
10. [VS Code Launch Configurations](#vs-code-launch-configurations)

---

## System Overview

### Purpose
This system generates **synthetic doctor-patient depression screening dialogues** using multi-agent LLM orchestration. It creates realistic conversations that:
- Follow DSM-5 depression screening protocols
- Reflect diverse personality types (based on Big Five personality traits)
- Include natural conversational dynamics (rapport building, follow-ups, disclosure gradients)
- Produce structured training data with ground truth labels
- Support patients across a wide age range (16-80) with age-appropriate contexts

### Core Innovation
Uses a **dual-manager architecture**:
1. **Doctor Manager**: Decides conversation strategy (DSM screening, follow-up questions, or rapport building)
2. **Patient Manager**: Controls patient response characteristics (disclosure level, directness, verbosity)

This creates more realistic, varied dialogues than single-agent approaches.

---

## Architecture

### Multi-Agent System

```
┌─────────────────────────────────────────────────────────┐
│                     Session Runner                       │
│         (Orchestrates entire conversation lifecycle)     │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       ┌──────────┐          ┌──────────┐
       │  Doctor  │          │  Patient │
       │  Agent   │          │  Agent   │
       │(Stateful)│          │(Stateful)│
       └────┬─────┘          └─────┬────┘
            │                      │
            │                      │
       ┌────┴─────┐          ┌─────┴────┐
       │  Doctor  │          │  Patient │
       │  Manager │          │  Manager │
       │(Stateless)│         │(Stateless)│
       └──────────┘          └──────────┘
```

### Agent Roles

**Doctor Agent (Stateful)**
- Conducts the interview
- Maintains conversation context across turns
- Follows guidance from Doctor Manager
- Persona-specific (8 different personas available)

**Doctor Manager (Stateless)**
- Strategic decision maker
- Decides next action: DSM, FOLLOW_UP, or RAPPORT
- Four modes: Normal, Low-Turns, Force-DSM, Post-DSM
- Ensures balanced coverage of DSM items with graceful transitions

**Patient Agent (Stateful)**
- Simulates depressed patient
- Responds based on personality profile
- Maintains consistency across conversation
- Follows guidance from Patient Manager

**Patient Manager (Stateless)**
- Controls response characteristics
- Manages disclosure gradient (MINIMIZE → PARTIAL → OPEN)
- Adjusts directness and verbosity
- Ensures symptom consistency with ground truth

---

## Key Components

### 1. Entry Point: `cli.py`

**Main Functions:**
- Parse command-line arguments
- Set up random seed for reproducibility
- Generate sessions with optional forced parameters
- Test modes for profile sampling and manager testing

**Command-Line Options:**
```bash
# Basic usage (uses gpt-5-mini by default)
python -m synthetic_datagen.cli

# With options
python -m synthetic_datagen.cli \
  --model gpt-5-mini \
  --seed 42 \
  --num-sessions 10 \
  --forced-template NEUROTICISM_HIGH \
  --forced-density LOW \
  --forced-persona warm_validating \
  --forced-age "16-19" \
  --forced-trust guarded \
  --forced-verbosity terse \
  --forced-expressiveness intense \
  --forced-modifiers "hostile,irritable"
```

**Test Modes:**
- `--test-profile`: Sample profile without calling LLM
- `--test-patient-manager`: Test patient manager with dummy input

**Forced Parameters (for testing edge cases):**
| Parameter | Description | Values |
|-----------|-------------|--------|
| `--forced-template` | Big Five template | NEUROTICISM_HIGH, EXTRAVERSION_LOW, etc. |
| `--forced-density` | Episode density | ULTRA_LOW, LOW, MED, HIGH |
| `--forced-persona` | Doctor persona | warm_validating, dismissive_rushed, etc. |
| `--forced-age` | Age range | 16-19, 20-24, ..., 70-80 |
| `--forced-trust` | Trust level | guarded, neutral, open |
| `--forced-verbosity` | Verbosity | terse, moderate, detailed |
| `--forced-expressiveness` | Expressiveness | flat, balanced, intense |
| `--forced-modifiers` | Trait modifiers | comma-separated list |

### 2. Configuration: `config.py`

**Logging Configuration:**

| Setting | Default | Description |
|---------|---------|-------------|
| `LOG_MINIMAL` | "minimal" | Just dialogue (doctor/patient responses) |
| `LOG_LIGHT` | "light" | Adds manager guidance, next action, disclosure state |
| `LOG_HEAVY` | "heavy" | Everything: tone tags, key points, emotional state, full instructions |
| `DEFAULT_LOG_LEVEL` | LOG_MINIMAL | Default verbosity for CLI |

**Model & Session Settings:**

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_MODEL` | gpt-4.1-mini | Default model for all agents |
| `DEFAULT_TEMPERATURE` | 1.0 | Temperature for generation |
| `MANAGER_TEMPERATURE` | 1.0 | Manager agent decisions |
| `BACKGROUND_WRITER_TEMPERATURE` | 1.0 | Background generation |
| `DEFAULT_NUM_SESSIONS` | 1 | Default number of sessions to generate |
| `NUM_DSM_ITEMS` | 9 | Standard DSM-5 depression criteria |
| `BUFFER_TURNS` | 5 | Extra turns for natural wrap-up after DSM coverage |

**Available Models:**
- `gpt-4o`
- `gpt-4.1`
- `gpt-4.1-mini` (default)
- `gpt-5-mini`

**Pacing Targets:**
```python
PACING_TARGETS = {
    "brisk": 1.5,  # turns per symptom
    "med": 2.0,
    "slow": 2.5,
}
```

**Extended Pacing Personas:**
```python
# These personas get +0.5 turns per symptom (max 3.0)
EXTENDED_PACING_PERSONAS = {"very_warm_chatty", "trauma_informed_slow"}
```

### 3. Data Pools: `data/pools.py`

**Pacing Levels:**
```python
PACING_LEVELS = ["LOW", "MED", "HIGH"]
```

**Context Domains (7 options):**
```python
GENERIC_CONTEXT_DOMAINS = [
    "work/role strain",
    "relationships strain",
    "health concern",
    "self-worth/identity strain",
    "general stress/no clear trigger",
    "major life transition",
    "grief/bereavement",
]
MAX_CONTEXT_DOMAINS = 2  # Patients get 0-2 domains
```

**Voice Style Options (Independent Sampling):**
```python
# Each dimension sampled independently → 81 combinations (3^4)
VOICE_STYLE_OPTIONS = {
    "verbosity": ["terse", "moderate", "detailed"],
    "expressiveness": ["flat", "balanced", "intense"],
    "trust": ["guarded", "neutral", "open"],
    "intellect": ["low-functioning", "moderate-functioning", "high-functioning"],
}

# Humor is a separate dimension with weighted sampling
PATIENT_HUMOR_LEVELS = ["none", "occasional", "frequent"]
# Weights: 60% none, 30% occasional, 10% frequent
```

**Personal Background Pools:**
```python
LIVING_SITUATION_POOL = ["alone", "with partner", "with family", "shared housing"]
WORK_ROLE_POOL = ["employed", "student", "caregiving role", "between roles"]
ROUTINE_STABILITY_POOL = ["stable routine", "variable routine"]
SUPPORT_LEVEL_POOL = ["low support", "moderate support", "high support"]
```

**Intensity Levels:**
```python
INTENSITY_LEVELS = ["LOW", "MED", "HIGH"]
# Weighted: 30% LOW, 50% MED, 20% HIGH
```

**Episode Density:**
```python
EPISODE_DENSITY_LEVELS = ["ULTRA_LOW", "LOW", "MED", "HIGH"]
# Weighted distribution: ~10%, 25%, 45%, 20%
```

**Age Ranges (9 brackets):**
```python
AGE_RANGES = [
    "16-19",  # Late teens
    "20-24",  # Early 20s
    "25-29",  # Late 20s
    "30-34",  # Early 30s
    "35-39",  # Late 30s
    "40-49",  # 40s
    "50-59",  # 50s
    "60-69",  # 60s
    "70-80",  # 70s+
]
```

**Age Weights by Work Role:**
```python
# Columns: 16-19, 20-24, 25-29, 30-34, 35-39, 40-49, 50-59, 60-69, 70-80
AGE_WEIGHTS_BY_ROLE = {
    "student":        [4, 5, 3, 1, 0.5, 0.3, 0.2, 0.1, 0.05],  # Young-heavy
    "employed":       [0.5, 2, 3, 3, 3, 2.5, 2, 1, 0.3],       # Working age
    "caregiving role": [0.1, 0.3, 0.8, 1.5, 2.5, 3, 3, 2.5, 2], # Older-heavy
    "between roles":  [1.5, 2, 2.5, 2, 1.5, 1, 0.8, 0.6, 0.4],  # Broad
}
```

**Life Facet Categories (67 categories across 11 domains):**
- Identity: identity_stage, cultural_background, sense_of_belonging, values, self_view
- Goals: short_term_goal, long_term_goal, stalled_goal, source_of_motivation
- People: partner, friend, family_pattern, work_ally, conflictual_relationship
- Work/Roles: work_pressure, achievement, role_conflicts, financial_pressure, responsibility_load
- Health: physical_health, sleep_pattern, diagnoses, help_seeking, body_image
- Stressors: primary_stressor, secondary_stressors, loss_or_change, unresolved_issue, fear_theme
- Coping: coping_style, routines, soothing_activities, unhelpful_coping, digital_habits
- Interests: hobbies, small_joys, quirks, self_presentation, competence_areas
- History: past_difficult_period, relationship_breakdown, family_history, significant_move
- Constraints: housing_feel, access_to_resources, time_energy_constraints
- Beliefs: explanatory_style, beliefs_about_help, self_worth_beliefs, hopes_for_future

**Trauma/Adversity Facets (gated):**
```python
TRAUMA_ADVERSITY_FACETS = [
    "past_difficult_period",
    "prior_relationship_disappointment_or_breakdown",
    "family_history_of_health_or_mental_health_issues",
    "significant_move_or_transition",
    "loss_or_change",
]
# Rules: Max 2 in required set, max 1 high-salience per patient
```

### 4. Templates: `data/templates.py`

**Big Five Depression Templates (8 templates):**

| Template | Affective Style | Cognitive Style | Emphasized Symptoms |
|----------|-----------------|-----------------|---------------------|
| NEUROTICISM_HIGH | Intense sadness, guilt, irritability | Rumination, self-blame, catastrophizing | Depressed mood, Fatigue, Guilt, Concentration |
| EXTRAVERSION_LOW | Emotional flatness, social withdrawal | Hopelessness, pessimism | Anhedonia, Psychomotor, Fatigue |
| CONSCIENTIOUSNESS_HIGH | Controlled affect, tension | Perfectionism, self-criticism | Weight/appetite, Sleep, Fatigue, Guilt |
| CONSCIENTIOUSNESS_LOW | Apathy, disengagement | Disorganization, forgetfulness | Psychomotor, Fatigue, Concentration |
| AGREEABLENESS_HIGH | Empathic sadness, guilt about others | Moral/relational rumination | Depressed mood, Sleep, Fatigue, Guilt |
| AGREEABLENESS_LOW | Irritability, anger, externalized blame | Defensive/hostile thoughts | Depressed mood, Psychomotor agitation, Sleep |
| OPENNESS_HIGH | Existential sadness | Philosophical rumination | Depressed mood, Anhedonia, Suicidal thoughts |
| OPENNESS_LOW | Constricted affect | Literal thinking, low insight | Weight/appetite, Sleep, Fatigue |

**Template Modifiers (trait nuances):**
Each template includes 3-5 modifiers sampled at generation time:
- NEUROTICISM_HIGH: worry-prone, self-blaming, catastrophizing, emotionally volatile, guilt-ridden
- EXTRAVERSION_LOW: socially withdrawn, emotionally flat, anhedonic, passive, isolated
- etc.

**Frequency Categories:**
```python
FREQUENCY_CATEGORIES = {
    "NONE": "Not at all",
    "RARE": "One or two days",
    "SOME": "Three to five days",
    "OFTEN": "Six to ten days"
}
```

### 5. Doctor Personas: `prompts/doctor_personas.py`

**8 Doctor Communication Styles:**

| Persona ID | Style | First Greeting Example |
|------------|-------|------------------------|
| warm_validating | Supportive, encouraging, normalizes feelings | "Hi, thanks for coming in today. How have things been going for you?" |
| neutral_efficient | Professional, focused, minimal small talk | "Good to meet you. Let's go ahead and discuss how you've been feeling." |
| gentle_brisk | Kind but moves conversation along | "Hello! Let's chat about how you've been lately." |
| matter_of_fact_kind | Direct yet respectful | "Hi there. I'm here to check in on how you've been doing recently." |
| trauma_informed_slow | Gentle, patient, careful with sensitive topics | "Welcome. Before we begin, I want you to know we can take this at whatever pace feels right." |
| structured_psychoeducational | Explains rationale for questions | "Hello. I'll be asking you some standard questions about your mood..." |
| time_pressed_clinical | Efficient, concise, redirects to symptoms | "Hello. I have a few focused questions to ask you today." |
| dismissive_rushed | Busy, slightly impatient, professional but rushed | "Okay, let's make this quick. What brings you in today?" |

**Microstyle Variation (per-session randomization):**
```python
# Each dimension sampled independently per session
"warmth": ["low", "med", "high"]
"directness": ["low", "med", "high"]
"pacing": ["slow", "med", "brisk"]
"humor": ["none", "light", "dry"]
"animation": ["reserved", "moderate", "animated"]
```

---

## Data Structures

### Patient Profile Structure

```python
{
    "template_id": "NEUROTICISM_HIGH",
    "template": {
        "affective": "...",
        "cognitive": "...",
        "somatic": "...",
        "emphasized_symptoms": [...],
        "specifier": "...",
        "modifiers": [...]
    },
    "personality": {
        "BIG5_TEMPLATE": "NEUROTICISM_HIGH",
        "AFFECTIVE_STYLE": "...",
        "COGNITIVE_STYLE": "...",
        "SOMATIC_STYLE": "...",
        "PACING": "MED",
        "CONTEXT_DOMAINS": ["work/role strain"],
        "EPISODE_DENSITY": "MED",
        "AGE_RANGE": "30-34",  # Sampled based on work role
        "VOICE_STYLE": {
            "verbosity": "moderate",
            "expressiveness": "balanced",
            "trust": "neutral",
            "intellect": "moderate-functioning",
            "humor": "none"  # Separate weighted sampling
        },
        "PERSONAL_BACKGROUND": {
            "living_situation": "alone",
            "work_role": "employed",
            "support_level": "moderate support"
        },
        "MODIFIERS": ["worry-prone", "self-blaming"],
        "EMPH_INTENSITY": {
            "Depressed mood": "MED",
            "Fatigue or loss of energy": "HIGH",
            ...
        },
        "EXTRA_ELEVATED_SYMPTOMS": []  # 0-1 non-emphasized symptoms elevated
    },
    "depression_profile": {
        "Depressed mood": "OFTEN",
        "Loss of interest or pleasure": "SOME",
        "Significant weight/appetite changes": "RARE",
        "Sleep disturbances": "OFTEN",
        "Psychomotor agitation or retardation": "NONE",
        "Fatigue or loss of energy": "OFTEN",
        "Feelings of worthlessness or excessive guilt": "OFTEN",
        "Difficulty concentrating or indecisiveness": "SOME",
        "Recurrent thoughts of death or suicide": "RARE"
    },
    "life_background": {  # From background writer
        "name": "Jordan",
        "age_range": "early 30s",
        "pronouns": "they/them",
        "core_roles": ["software developer", "part-time caregiver"],
        "core_relationships": [...],
        "core_stressor_summary": "...",
        "life_facets": [...]
    },
    "background_writer_prompt_trace": {
        "agent": "background_writer",
        "system_prompt": "...",
        "input": "...",
        "output": "..."
    }
}
```

### Session Output Structure

```python
{
    "run_timestamp_utc": "2024-...",
    "agent_id": "AGENT_<hash>",
    "template_id": "NEUROTICISM_HIGH",
    "template_details": {...},
    "persona_id": "warm_validating",
    "persona_details": {...},
    "microstyle": {"warmth": "med", "directness": "high", "pacing": "med"},
    "personality": {...},
    "depression_profile_ground_truth": {...},
    "life_background": {...},  # Rich background from writer
    "asked_question_order": ["Depressed mood", "Sleep disturbances", ...],
    "doctor_manager_decisions": [
        {
            "turn": 1,
            "next_action": "DSM",
            "reason": "...",
            "doctor_instruction": "...",  # Guides doctor response
            "dsm_symptom_key": "Depressed mood"
        },
        ...
    ],
    "patient_manager_decisions": [
        {
            "turn": 1,
            "doctor_move": "DSM",
            "guidance": {
                "directness": "MED",
                "disclosure_stage": "PARTIAL",
                "target_length": "MEDIUM",
                "emotional_state": "neutral",  # Can be anxious, tearful, etc.
                "tone_tags": ["cooperative", "somewhat-guarded"],
                "patient_instruction": "..."  # Guides patient response
            }
        },
        ...
    ],
    "final_disclosure_state": "OPEN",
    "conversation": [
        {"speaker": "doctor", "text": "Hello! How are you doing?"},
        {"speaker": "patient", "text": "I'm okay, I guess."},
        ...
    ],
    "token_usage": {
        "doctor": {...},
        "patient": {...},
        "doctor_manager": {...},
        "patient_manager": {...},
        "background_writer": {...}
    }
}
```

---

## Prompt Engineering

### Doctor Base Prompt (`prompts/doctor_base.py`)

Foundation for all doctor agents. Includes:
- Role definition
- Directive tag system (`<NEXT_QUESTION>`, `<FOLLOW_UP>`, `<RAPPORT>`)
- Conversation guidelines
- DSM symptom coverage instructions
- Natural conversation principles
- Patient snapshot (if life_background available)

### Patient Prompt (`prompts/patient_prompt.py`)

Built dynamically per session:

```python
def build_patient_system_prompt(personality, depression_profile, life_background=None):
    """
    Constructs full patient persona including:
    - Age range (always present)
    - Big Five template details
    - Modifiers (trait nuances)
    - DSM symptom profile
    - Voice style parameters (including humor)
    - Personal background or rich life background
    - Response pacing instructions
    - Identity section (if life_background present)
    - Key life facets (2-4 high/med salience)
    """
```

**Key Sections:**
1. Big Five Depression Template (including age)
2. Template Modifiers
3. DSM-5 Depression Symptom Profile
4. Current Life Context (Broad Domains)
5. Your Identity (if life_background)
6. Key Life Details (high-salience facets)
7. Roleplay Instructions
8. Your Voice Style
9. Authenticity Guidelines

### Doctor Manager Prompts (`prompts/doctor_manager_prompt.py`)

**Four Modes:**

1. **NORMAL MODE**: Balanced decision-making between DSM, FOLLOW_UP, and RAPPORT
2. **LOW-TURNS MODE**: Prioritizes DSM when behind schedule (ratio < target turns/symptom)
3. **FORCE-DSM MODE**: Forces DSM when remaining_turns ≤ remaining_dsm, but provides full persona-aware transition guidance (not abrupt)
4. **POST-DSM MODE**: Handles wrap-up after all DSM items covered (FOLLOW_UP, RAPPORT, or END)

**Output Format:**
```json
{
  "next_action": "FOLLOW_UP|RAPPORT|DSM",
  "reason": "...",
  "doctor_instruction": "...",
  "dsm_symptom_key": "..."
}
```

### Patient Manager Prompt (`prompts/patient_manager_prompt.py`)

**Output Format:**
```json
{
  "directness": "LOW|MED|HIGH",
  "disclosure_stage": "MINIMIZE|PARTIAL|OPEN",
  "target_length": "SHORT|MEDIUM|LONG",
  "emotional_state": "neutral|anxious|tearful|flat|irritable|hopeful",
  "tone_tags": ["tag1", "tag2"],
  "key_points_to_reveal": ["..."],
  "key_points_to_avoid": ["..."],
  "patient_instruction": "..."
}
```

**Emotional State Values:**
- `neutral` - Default calm state (most common)
- `anxious` - Worried, nervous, uneasy
- `tearful` - Close to tears or crying
- `flat` - Emotionally blunted, low affect
- `irritable` - Frustrated, annoyed, short-tempered
- `hopeful` - Cautiously optimistic (typically late in conversation)

### Background Writer Prompt (`prompts/background_writer_prompt.py`)

**Instructions Include:**
- Must use provided age_range exactly
- Age-appropriate facet guidance for each life stage
- Trauma/adversity gating rules
- JSON-only output format
- Consistency requirements with depression profile

**Age-Appropriate Facet Guidance:**
- Teenagers (16-19): school pressures, friend drama, parental relationships, emerging identity
- Young adults (20-34): career building, romantic partnerships, financial independence
- Middle adults (35-54): career peaks/plateaus, parenting, aging parents, work-life balance
- Older adults (55-69): empty nest, pre-retirement, grandchildren, legacy
- Seniors (70-80): retirement adjustment, health decline, isolation, end-of-life reflection

---

## Workflow

### Session Generation Flow

```
1. INITIALIZATION
   ├─ Parse CLI arguments
   ├─ Set random seed
   └─ Load configuration

2. PROFILE GENERATION
   ├─ Select Big Five template
   ├─ Sample personality traits
   ├─ Sample work role → age range (role-weighted)
   ├─ Sample voice style (independent dimensions)
   ├─ Generate depression profile
   ├─ Call background writer (generates life_background)
   └─ Select doctor persona

3. AGENT SETUP
   ├─ Build doctor system prompt (with patient snapshot)
   ├─ Build patient system prompt (with life_background)
   ├─ Initialize stateful agents
   └─ Set disclosure state (based on trust)

4. CONVERSATION LOOP (until max turns)
   │
   ├─ DOCTOR MANAGER PHASE
   │  ├─ Build manager input (full conversation history)
   │  ├─ Call doctor manager (stateless)
   │  ├─ Parse decision (FOLLOW_UP|RAPPORT|DSM)
   │  └─ Generate doctor_instruction
   │
   ├─ DOCTOR AGENT PHASE
   │  ├─ Receive instruction with directive tags
   │  ├─ Generate doctor utterance (stateful)
   │  └─ Append to conversation
   │
   ├─ PATIENT MANAGER PHASE
   │  ├─ Build manager input (conversation + profile)
   │  ├─ Call patient manager (stateless)
   │  ├─ Parse guidance (disclosure, directness, length)
   │  └─ Update disclosure state
   │
   └─ PATIENT AGENT PHASE
      ├─ Receive guidance block (with patient_instruction)
      ├─ Generate patient response (stateful)
      └─ Append to conversation

5. POST-DSM PHASE (when all DSM items covered)
   ├─ Switch to post-DSM manager
   ├─ Continue with FOLLOW_UP/RAPPORT/END actions
   └─ Natural conversation wrap-up

6. OUTPUT GENERATION
   ├─ Save transcript (JSON)
   ├─ Save raw log with full details
   ├─ Save prompt traces (including background_writer)
   └─ Report token usage
```

### Turn Budget Management

**Pacing-Based Budget:**
```python
target_turns_per_symptom = PACING_TARGETS[microstyle_pacing]
# brisk: 1.5, med: 2.0, slow: 2.5

if persona in EXTENDED_PACING_PERSONAS:
    target_turns_per_symptom = min(target_turns_per_symptom + 0.5, 3.0)

base_turn_budget = ceil(target_turns_per_symptom * NUM_DSM_ITEMS)
buffer_turns = 5  # Extra turns for rapport, follow-up, and natural conclusion
max_doctor_turns = base_turn_budget + buffer_turns + 1
```

**Manager Mode Selection (in order of priority):**
1. **Post-DSM Mode**: When `remaining_dsm_items == 0` (all DSM covered)
2. **Force-DSM Mode**: When `remaining_turns <= remaining_dsm_items` (no slack left)
   - Uses FORCE_DSM prompt to get smooth transition guidance
   - Doctor still gets full persona-aware instructions
3. **Low-Turns Mode**: When `ratio < target_turns_per_symptom` (behind schedule)
4. **Normal Mode**: Otherwise (on schedule or ahead)

**Low-Turns Trigger:** When `ratio < target_turns_per_symptom`, switches to low-turns mode (DSM preferred but not forced).

---

## Output Structure

### Three Output Files Per Session

1. **Transcript** (`outputs/transcripts/transcript_AGENT_<hash>.json`)
   - Clean conversation format
   - Metadata and ground truth
   - Manager decisions
   - Token usage

2. **Raw Log** (`outputs/logs/session_AGENT_<hash>_raw.json`)
   - Full session data
   - All prompt traces
   - Complete debugging info

3. **Prompt Trace** (`outputs/prompt_traces/prompttrace_AGENT_<hash>.json`)
   - All agent inputs/outputs
   - System prompts used
   - Turn-by-turn trace
   - Background writer trace

### Agent ID Generation

The agent ID is a SHA256 hash of all major generation levers, ensuring unique IDs when any parameter changes:

```python
profile_key_parts = [
    # Patient levers (10)
    f"TEMPLATE={template_id}",
    f"P_PACING={personality.get('PACING', 'MED')}",
    f"DENSITY={personality.get('EPISODE_DENSITY', 'MED')}",
    f"AGE={personality.get('AGE_RANGE', '')}",
    f"TRUST={voice_style.get('trust', 'neutral')}",
    f"VERBOSITY={voice_style.get('verbosity', 'moderate')}",
    f"EXPRESSIVENESS={voice_style.get('expressiveness', 'balanced')}",
    f"INTELLECT={voice_style.get('intellect', 'average')}",
    f"P_HUMOR={voice_style.get('humor', 'none')}",
    f"MODS={','.join(sorted(modifiers))}",
    # Doctor levers (6)
    f"PERSONA={doctor_persona_id}",
    f"D_WARMTH={doctor_microstyle.get('warmth', 'med')}",
    f"D_DIRECT={doctor_microstyle.get('directness', 'med')}",
    f"D_PACING={doctor_microstyle.get('pacing', 'med')}",
    f"D_HUMOR={doctor_microstyle.get('humor', 'none')}",
    f"D_ANIMATION={doctor_microstyle.get('animation', 'med')}",
]
# Plus all 9 depression symptom frequencies
for symptom in sorted(depression_profile.keys()):
    profile_key_parts.append(f"{symptom}={depression_profile[symptom]}")

profile_key = "|".join(profile_key_parts)
agent_id = f"AGENT_{hashlib.sha256(profile_key.encode()).hexdigest()[:16]}"
```

**Included Levers:**
- **Patient (10):** Template, pacing, density, age, trust, verbosity, expressiveness, intellect, humor, modifiers
- **Doctor (6):** Persona ID, warmth, directness, pacing, humor, animation microstyles
- **Depression Profile (9):** All symptom frequencies

Deterministic based on all profile parameters for reproducibility - same parameters always produce the same agent_id.

---

## How to Extend

### Adding New Doctor Personas

**File:** `synthetic_datagen/prompts/doctor_personas.py`

```python
DOCTOR_PERSONAS.append({
    "id": "new_persona_name",
    "first_greeting": "Your opening line here",
    "system_prompt": """
Persona style:
- Describe communication style
- Tone characteristics
- Approach to questioning
""",
})
```

### Adding New Big Five Templates

**File:** `synthetic_datagen/data/templates.py`

```python
BIG5_DEP_TEMPLATES["NEW_TEMPLATE_NAME"] = {
    "affective": "Emotional presentation style",
    "cognitive": "Thinking patterns",
    "somatic": "Physical symptom patterns",
    "emphasized_symptoms": [
        "List of 3-4 DSM symptoms to emphasize"
    ],
    "specifier": "DSM-5 specifier (e.g., 'MDD with anxious distress')",
    "modifiers": [
        "trait-modifier-1",
        "trait-modifier-2",
        "trait-modifier-3"
    ]
}
```

### Adding New Age Ranges

**File:** `synthetic_datagen/data/pools.py`

```python
# Add to AGE_RANGES list
AGE_RANGES.append("80-90")

# Update all weight lists in AGE_WEIGHTS_BY_ROLE
AGE_DEFAULT_WEIGHTS.append(0.1)
```

### Adding New Voice Style Dimensions

**File:** `synthetic_datagen/data/pools.py`

```python
VOICE_STYLE_OPTIONS["new_dimension"] = ["low", "med", "high"]
```

Update `profile_generation.py` and `patient_prompt.py` to handle the new dimension.

### Adding New Life Facet Categories

**File:** `synthetic_datagen/data/pools.py`

```python
LIFE_FACET_CATEGORIES.append("new_facet_category")
```

Update `background_writer.py` to handle the new category in weighting.

---

## Configuration

### Environment Variables

**Required:**
```bash
OPENAI_API_KEY=your_api_key_here
```

Set via `.env` file or export directly.

### Model Selection

```bash
python -m synthetic_datagen.cli --model gpt-5-mini  # Default
python -m synthetic_datagen.cli --model gpt-4o
python -m synthetic_datagen.cli --model gpt-4.1
python -m synthetic_datagen.cli --model gpt-4.1-mini
```

### Random Seed

```bash
python -m synthetic_datagen.cli --seed 42  # Reproducible
python -m synthetic_datagen.cli            # Random seed generated
```

### Forced Parameters (Testing)

```bash
# Force template
--forced-template NEUROTICISM_HIGH

# Force episode density
--forced-density ULTRA_LOW

# Force doctor persona
--forced-persona dismissive_rushed

# Force age range
--forced-age "70-80"

# Force voice style dimensions
--forced-trust guarded
--forced-verbosity terse
--forced-expressiveness intense

# Force modifiers
--forced-modifiers "hostile,irritable,avoidant"
```

---

## VS Code Launch Configurations

Pre-configured launch profiles in `.vscode/launch.json`:

### Standard Configurations

| Name | Description |
|------|-------------|
| Run with GPT-4o | Use GPT-4o model |
| Run with GPT-4.1 | Use GPT-4.1 model |
| Run with GPT-4.1-mini | Use GPT-4.1-mini model |
| Run with GPT-5-mini | Use GPT-5-mini model (default) |
| Test Profile Generation | Sample profile without LLM |
| Test Patient Manager | Test patient manager with dummy input |
| Run with Custom Args | Empty args for manual configuration |

### Test Configurations (Edge Cases)

| Name | Purpose | Key Args |
|------|---------|----------|
| TEST: Very Old Patient (70-80) | Elderly patient simulation | `--forced-age 70-80` |
| TEST: Very Young Patient (16-19) | Teen patient simulation | `--forced-age 16-19` |
| TEST: Angry/Hostile Patient | High expressiveness + hostile | `--forced-expressiveness intense`, `--forced-modifiers hostile,irritable`, `--forced-template NEUROTICISM_HIGH` |
| TEST: Dismissive Doctor | Dismissive/rushed doctor persona | `--forced-persona dismissive_rushed` |
| TEST: Patient Who Hates Doctors | Guarded + hostile patient | `--forced-trust guarded`, `--forced-modifiers hostile,avoidant` |
| TEST: Terse Elderly Caregiver | Short responses, flat affect | `--forced-age 60-69`, `--forced-verbosity terse`, `--forced-expressiveness flat` |
| TEST: Low Intellect Patient | Lower cognitive functioning | `--forced-template CONSCIENTIOUSNESS_LOW`, `--forced-modifiers cognitively_slow,self_deprecating` |
| TEST: Very Detailed/Talkative Patient | Maximum verbosity | `--forced-verbosity detailed`, `--forced-expressiveness intense`, `--forced-trust open`, `--forced-density HIGH` |
| TEST: Minimal Symptoms (ULTRA_LOW) | Near-zero symptom load | `--forced-density ULTRA_LOW` |
| TEST: Severe Depression (HIGH density) | Maximum symptom severity | `--forced-density HIGH`, `--forced-template NEUROTICISM_HIGH` |
| TEST: Warm Doctor + Guarded Patient | Interaction stress test | `--forced-persona warm_validating`, `--forced-trust guarded` |
| TEST: Perfectionist Middle-Aged | Conscientious in 40s | `--forced-template CONSCIENTIOUSNESS_HIGH`, `--forced-age 40-49`, `--forced-modifiers perfectionist,self_critical` |

---

## Advanced Topics

### Depression Profile Generation

**ULTRA_LOW Mode:**
- Allows 0-2 total symptoms (vs normal emphasis-based distribution)
- Useful for subclinical or minimal symptom cases
- Maintains emphasized symptom priority when selecting
- Target count: random choice of 0, 1, or 2

**Standard Density Modes:**
- LOW: ~7/10 NONE, ~2/10 RARE, ~1/10 SOME for non-emphasized
- MED: ~5/10 NONE, ~3/10 RARE, ~2/10 SOME
- HIGH: ~3/10 NONE, ~3/10 RARE, ~4/10 SOME

**Emphasized Symptom Intensity:**
```python
# Each emphasized symptom gets intensity (LOW/MED/HIGH)
if intensity == "LOW":
    weights = [5, 4, 1]  # RARE, SOME, OFTEN
elif intensity == "HIGH":
    weights = [2, 8]     # SOME, OFTEN (no RARE)
else:  # MED
    weights = [5, 4, 1]  # SOME, OFTEN, RARE order
```

### Disclosure Gradient System

**Initialization (based on trust):**
```python
if trust == "guarded":
    initial_disclosure = "MINIMIZE"
elif trust == "open":
    initial_disclosure = "OPEN"
else:  # neutral
    initial_disclosure = "PARTIAL"
```

**Evolution:**
- Patient Manager adjusts disclosure_stage each turn
- FOLLOW_UP and RAPPORT from supportive doctors can increase disclosure
- Guarded patients stay MINIMIZE longer
- Progression: MINIMIZE → PARTIAL → OPEN

### Life Facets System

**Background Writer Process:**
1. Compute symptom severity (minimal/mild/moderate/severe)
2. Select 5-8 required facet categories based on:
   - Context domain relevance (boosted weights)
   - Severity level (trauma gating)
   - Core stressor inclusion (always high weight)
3. Call LLM with structured prompt
4. Parse JSON output into PatientLifeBackground

**Severity Computation:**
```python
significant_count = SOME + OFTEN symptoms
often_count = OFTEN symptoms

if significant_count == 0:
    severity = "minimal"
elif significant_count <= 2 or (significant_count <= 4 and often_count == 0):
    severity = "mild"
elif significant_count <= 5 or often_count <= 3:
    severity = "moderate"
else:
    severity = "severe"
```

**Trauma Gating:**
- Mild cases: trauma weights * 0.5
- Moderate/severe: trauma weights * 1.5
- Max 2 trauma facets in required set
- Max 1 high-salience trauma per patient

### Token Usage Tracking

Tracked separately for each agent:
```python
token_usage = {
    "doctor": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    "patient": {...},
    "doctor_manager": {...},
    "patient_manager": {...},
    "background_writer": {...}  # ~1000-1400 tokens per session
}
```

### Stateful vs Stateless Agents

**Stateful (Doctor, Patient):**
- Maintain conversation history in SQLiteSession
- Context grows with each turn
- Natural memory of prior exchanges

**Stateless (Managers, Background Writer):**
- Fresh session each call
- Receive full context in input
- Ensures consistent strategic view
- Prevents context drift

---

## Troubleshooting

### Common Issues

**1. Invalid JSON from Manager**
- Fallback logic automatically handles this
- Defaults to safe actions (DSM or FOLLOW_UP)
- Check `parse_doctor_manager_output()` and `parse_patient_manager_output()`

**2. DSM Coverage Incomplete**
- Check `max_doctor_turns` calculation
- Verify DSM coverage guard is triggering
- Review low-turns mode threshold

**3. Inconsistent Patient Responses**
- Check `depression_profile` mapping in patient manager input
- Verify voice_style parameters are being used
- Review disclosure state management

**4. Background Writer Failures**
- Falls back gracefully to basic background
- Check JSON parsing in `parse_background_writer_output()`
- Review prompt trace for errors

**5. Age Mismatch**
- Ensure `age_range` is in personality dict
- Check background writer output matches input age
- Verify patient prompt includes age

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspecting Outputs

**Prompt Traces:** Review `outputs/prompt_traces/` to see exact inputs/outputs for each agent.

**Manager Decisions:** Check `doctor_manager_decisions` and `patient_manager_decisions` in transcript.

**Raw Logs:** Full debugging info in `outputs/logs/`.

---

## Best Practices

### For Generation
1. Use consistent seeds for reproducible testing
2. Test with `--test-profile` before full runs
3. Gradually adjust parameters (don't change everything at once)
4. Review prompt traces to understand agent behavior

### For Extension
1. Follow existing naming conventions
2. Add new items to appropriate data files
3. Test with forced parameters first
4. Document custom additions

### For Research
1. Track agent_id for reproducibility
2. Save generation parameters with results
3. Use diverse seeds for dataset variety
4. Monitor token usage for cost management

---

## Summary

This system provides a sophisticated framework for generating realistic depression screening dialogues through:

1. **Multi-Agent Orchestration**: Doctor, Patient, and Manager agents work together
2. **Personality-Based Variation**: Big Five templates create diverse presentations
3. **Age-Aware Generation**: Role-based age sampling with age-appropriate contexts
4. **Independent Voice Dimensions**: 81+ voice style combinations (3^4 + humor)
5. **Rich Life Backgrounds**: LLM-generated life contexts with facet system
6. **Natural Conversation Flow**: Manager agents ensure realistic progression
7. **Ground Truth Labels**: Every session includes full symptom profile
8. **Extensive Configurability**: Easy to extend and customize

The modular architecture makes it straightforward to add new personas, templates, or behavioral patterns while maintaining system coherence.

---

**Generated:** 2024
**Version:** 5.3
**Default Model:** gpt-4.1-mini
