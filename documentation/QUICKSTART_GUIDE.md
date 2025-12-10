# Quick Start Guide: Running the Synthetic Data Generation System

## Prerequisites

1. **Python Environment**: Python 3.8+ installed
2. **Dependencies**: Install required packages
3. **OpenAI API Key**: Required for full session generation

## Setup Instructions

### 1. Install Dependencies

```bash
# Install the agents library and other dependencies
pip install agents openai
```

### 2. Set Your OpenAI API Key

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Running the Code

### Option 1: Test Without API Key (No LLM Calls)

Test profile generation without using API credits:

```bash
python -m synthetic_datagen.cli --test-profile --seed 42
```

**What it does:**
- Samples a random patient profile
- Shows template, personality traits, depression symptoms
- Displays voice style, pacing, episode density
- **No API calls or costs**

**Example output:**
```json
{
  "template_id": "EXTRAVERSION_LOW",
  "personality": {
    "VOICE_STYLE": {
      "verbosity": "terse",
      "expressiveness": "intense",
      "trust": "open",
      "intellect": "high-functioning"
    },
    "PACING": "HIGH",
    ...
  },
  "depression_profile": {
    "Depressed mood": "SOME",
    "Loss of interest or pleasure": "OFTEN",
    ...
  }
}
```

---

### Option 2: Test Patient Manager (Requires API Key)

Test the patient manager in isolation:

```bash
# Set API key first (see Setup Instructions above)
python -m synthetic_datagen.cli --test-patient-manager --seed 42
```

**What it does:**
- Samples a patient profile
- Creates a dummy doctor question
- Calls the patient manager to decide HOW patient should respond
- Shows the behavioral guidance (directness, disclosure, tone, etc.)
- **Uses ~300 tokens (~$0.01 with GPT-4)**

**Example output:**
```
=== TEST PATIENT MANAGER MODE ===
Testing patient manager with dummy scenario...

Template: EXTRAVERSION_LOW
Voice style: {'verbosity': 'terse', 'expressiveness': 'intense', 'trust': 'open', 'intellect': 'high-functioning'}
Initial disclosure state: OPEN

=== Patient Manager Raw Output ===
{
  "directness": "HIGH",
  "disclosure_stage": "OPEN",
  "target_length": "MEDIUM",
  "tone_tags": ["cooperative", "matter-of-fact"],
  "key_points_to_reveal": ["acknowledge some difficulty"],
  "key_points_to_avoid": ["specific examples"],
  "response_instruction": "Answer directly and acknowledge the question."
}

✓ Patient manager test completed successfully.
```

---

### Option 3: Generate Full Session (Requires API Key)

Generate a complete doctor-patient conversation:

```bash
# Set API key first (see Setup Instructions above)
python -m synthetic_datagen.cli --num-sessions 1 --seed 42
```

**What it does:**
- Generates a complete multi-turn conversation
- Doctor asks DSM-5 depression screening questions
- Patient responds with guidance from patient manager
- Saves transcript as JSON file
- **Uses ~5,000-10,000 tokens per session (~$0.15-0.30 with GPT-4)**

**Example output:**
```
Using random seed: 42

=== GENERATING 1 SESSION(S) ===

--- Session 1/1 ---

Doctor's first greeting: Hello, how are you doing today?
Patient reply (turn 1): I'm okay, I guess.

[Manager decision for doctor turn 1]
Action: DSM, Reason: Start with first DSM question

Doctor reply (turn 1): Over the past two weeks, how often have you felt down or hopeless?

[Patient manager decision for patient turn 1]
Guidance: directness=MED, disclosure=PARTIAL, length=MEDIUM

s (turn 1): Sometimes... I've had some rough days.

...

✓ Session saved: transcript_AGENT_abc123def456.json
  Template: EXTRAVERSION_LOW
  Persona: warm_validating
  Agent ID: AGENT_abc123def456
```

---

## Advanced Options

### Generate Multiple Sessions

```bash
python -m synthetic_datagen.cli --num-sessions 5
```

### Force Specific Template

```bash
python -m synthetic_datagen.cli --num-sessions 1 --forced-template NEUROTICISM_HIGH
```

Available templates:
- `NEUROTICISM_HIGH`
- `NEUROTICISM_LOW`
- `EXTRAVERSION_HIGH`
- `EXTRAVERSION_LOW`
- `OPENNESS_HIGH`
- `OPENNESS_LOW`
- `AGREEABLENESS_HIGH`
- `AGREEABLENESS_LOW`
- `CONSCIENTIOUSNESS_HIGH`
- `CONSCIENTIOUSNESS_LOW`

### Force Episode Density

```bash
python -m synthetic_datagen.cli --num-sessions 1 --forced-density HIGH
```

Options: `ULTRA_LOW`, `LOW`, `MED`, `HIGH`

### Log Level (Terminal Output Verbosity)

```bash
python -m synthetic_datagen.cli --num-sessions 1 --log-level minimal
```

Options:
- `minimal` - Just dialogue (doctor/patient responses and turn counter)
- `light` - Default. Adds manager guidance, next action, disclosure state, instructions
- `heavy` - Everything: tone tags, key points to reveal/avoid, emotional state, full instructions

**Example minimal output:**
```
--- Turn 1 ---
DOCTOR: Hello, how are you doing today?
PATIENT: I've been feeling down lately...
```

**Example light output (default):**
```
--- Doctor Turn 1 [NORMAL] (ratio=2.00, target=2.0) ---
  Manager: DSM → Depressed mood
  Reason: Start with first DSM question
  Instruction: Ask about depressed mood in a warm, conversational way.

  Doctor: Over the past two weeks, how often have you felt down...

--- Patient Turn 1 ---
  Disclosure: PARTIAL | Length: MEDIUM
  Instruction: Answer with some hesitation but acknowledge the feeling.
```

### Force Doctor Persona

```bash
python -m synthetic_datagen.cli --num-sessions 1 --forced-persona clinical_efficient
```

Available personas:
- `warm_validating`
- `clinical_efficient`
- `empathetic_probing`

### Combine Options

```bash
python -m synthetic_datagen.cli --num-sessions 3 --forced-template NEUROTICISM_HIGH --forced-density HIGH --seed 123
```

---

## Understanding the Output

### Transcript Files

Generated transcripts are saved as JSON files: `transcript_AGENT_<hash>.json`

**Location:** Current directory

**Contents:**
```json
{
  "run_timestamp_utc": "2025-11-24T23:50:00.000000",
  "agent_id": "AGENT_abc123def456",
  "template_id": "EXTRAVERSION_LOW",
  "personality": { ... },
  "depression_profile_ground_truth": { ... },
  "patient_manager_decisions": [
    {
      "turn": 1,
      "doctor_move": "DSM",
      "guidance": {
        "directness": "MED",
        "disclosure_stage": "PARTIAL",
        "target_length": "MEDIUM",
        "tone_tags": ["cooperative"],
        ...
      }
    }
  ],
  "final_disclosure_state": "OPEN",
  "conversation": [
    {"role": "assistant", "content": "Hello, how are you doing today?"},
    {"role": "user", "content": "I'm okay, I guess."},
    ...
  ]
}
```

### Key Fields

- **`depression_profile_ground_truth`**: True symptom frequencies (NONE/RARE/SOME/OFTEN)
- **`patient_manager_decisions`**: Behavioral guidance for each patient turn
- **`final_disclosure_state`**: How open the patient became (MINIMIZE/PARTIAL/OPEN)
- **`conversation`**: Complete dialogue transcript

---

## Troubleshooting

### "Error: OPENAI_API_KEY environment variable not set"

**Solution:** Set your API key (see Setup Instructions above)

### "Module not found" errors

**Solution:** Install dependencies:
```bash
pip install agents openai
```

### Import errors

**Solution:** Make sure you're in the project root directory:
```bash
cd c:\Users\artem\OneDrive\UNIVERSITY\Research\Depression_Detection
```

### Out of API credits

**Solution:** 
- Use `--test-profile` to test without API calls
- Check your OpenAI account balance
- Use a smaller number of sessions

---

## Cost Estimates (GPT-4)

- **Test Profile**: $0 (no API calls)
- **Test Patient Manager**: ~$0.01 per test
- **Full Session**: ~$0.15-0.30 per session
- **10 Sessions**: ~$1.50-3.00

*Costs are approximate and depend on conversation length*

---

## Examples

### Quick Test (No API Key)
```bash
# Just see what profiles look like
python -m synthetic_datagen.cli --test-profile
```

### Budget-Friendly Testing
```bash
# Test patient manager once
python -m synthetic_datagen.cli --test-patient-manager --seed 42
```

### Production Run
```bash
# Generate 10 diverse sessions
python -m synthetic_datagen.cli --num-sessions 10
```

### Controlled Generation
```bash
# Generate 5 high-neuroticism patients with high symptom density
python -m synthetic_datagen.cli --num-sessions 5 --forced-template NEUROTICISM_HIGH --forced-density HIGH --seed 999
```

---

## Next Steps

1. **Start Simple**: Run `--test-profile` to see profiles without API costs
2. **Test Manager**: Run `--test-patient-manager` to see behavioral guidance
3. **Generate One**: Run `--num-sessions 1` to create your first full conversation
4. **Scale Up**: Generate multiple sessions for your research dataset

## Need Help?

- Check `CHANGESET_P_SUMMARY.md` for implementation details
- Check `TEST_RESULTS_CHANGESET_P.md` for test results
- Review the code in `synthetic_datagen/` directory
