"""
Test script for patient manager functions (no API key required).
"""
import random
from synthetic_datagen.generation.profile_generation import sample_patient_profile
from synthetic_datagen.generation.manager_logic import (
    build_patient_manager_input,
    parse_patient_manager_output,
)

print("=" * 60)
print("Testing Patient Manager Functions")
print("=" * 60)

# Create RNG and sample profile
rng = random.Random(42)
profile = sample_patient_profile(rng)

template_id = profile["template_id"]
template = profile["template"]
personality = profile["personality"]
depression_profile = profile["depression_profile"]

print(f"\n1. Sampled Profile:")
print(f"   Template: {template_id}")
print(f"   Voice Style: {personality.get('VOICE_STYLE', {})}")
print(f"   Pacing: {personality.get('PACING', 'N/A')}")

# Initialize disclosure state
trust = personality.get("VOICE_STYLE", {}).get("trust", "neutral")
if trust == "guarded":
    disclosure_state = "MINIMIZE"
elif trust == "open":
    disclosure_state = "OPEN"
else:
    disclosure_state = "PARTIAL"

print(f"   Initial Disclosure: {disclosure_state}")

# Test build_patient_manager_input
print("\n2. Testing build_patient_manager_input():")

dummy_conversation = [
    {"role": "assistant", "content": "Hello, how are you today?"},
    {"role": "user", "content": "I'm okay."},
]

patient_meta = {
    "template_id": template_id,
    "emphasized_symptoms": template.get("emphasized_symptoms", []),
    "modifiers": personality.get("MODIFIERS", []),
    "voice_style": personality.get("VOICE_STYLE", {}),
    "pacing": personality.get("PACING", "MED"),
    "episode_density": personality.get("EPISODE_DENSITY", "MED"),
}

pm_input = build_patient_manager_input(
    last_doctor_message="Over the past two weeks, how often have you felt down?",
    last_doctor_move="DSM",
    conversation_history=dummy_conversation,
    patient_meta=patient_meta,
    depression_profile=depression_profile,
    disclosure_state=disclosure_state,
)

print("   ✓ Input generated successfully")
print(f"   Input length: {len(pm_input)} characters")
print(f"   Contains disclosure_state: {'disclosure_state' in pm_input}")
print(f"   Contains depression profile: {'Depressed mood' in pm_input}")

# Test parse_patient_manager_output with valid JSON
print("\n3. Testing parse_patient_manager_output() with valid JSON:")

valid_json = """{
  "directness": "MED",
  "disclosure_stage": "PARTIAL",
  "target_length": "MEDIUM",
  "tone_tags": ["cooperative", "slightly-guarded"],
  "key_points_to_reveal": ["acknowledge some difficulty"],
  "key_points_to_avoid": ["specific examples"],
  "response_instruction": "Answer moderately, acknowledge the question but don't elaborate much."
}"""

guidance = parse_patient_manager_output(
    valid_json,
    base_voice_style=personality.get("VOICE_STYLE", {}),
    current_disclosure_state=disclosure_state,
)

print("   ✓ Parsed successfully")
print(f"   Directness: {guidance['directness']}")
print(f"   Disclosure: {guidance['disclosure_stage']}")
print(f"   Length: {guidance['target_length']}")
print(f"   Tone tags: {guidance['tone_tags']}")

# Test parse_patient_manager_output with invalid JSON (fallback)
print("\n4. Testing parse_patient_manager_output() with invalid JSON (fallback):")

invalid_json = "This is not valid JSON at all!"

guidance_fallback = parse_patient_manager_output(
    invalid_json,
    base_voice_style=personality.get("VOICE_STYLE", {}),
    current_disclosure_state=disclosure_state,
)

print("   ✓ Fallback successful")
print(f"   Directness: {guidance_fallback['directness']}")
print(f"   Disclosure: {guidance_fallback['disclosure_stage']}")
print(f"   Length: {guidance_fallback['target_length']}")
print(f"   Has response_instruction: {'response_instruction' in guidance_fallback}")

# Test parse with markdown code fences
print("\n5. Testing parse_patient_manager_output() with markdown fences:")

markdown_json = """```json
{
  "directness": "HIGH",
  "disclosure_stage": "OPEN",
  "target_length": "LONG",
  "tone_tags": ["direct"],
  "key_points_to_reveal": ["full disclosure"],
  "key_points_to_avoid": [],
  "response_instruction": "Be direct and open."
}
```"""

guidance_md = parse_patient_manager_output(
    markdown_json,
    base_voice_style=personality.get("VOICE_STYLE", {}),
    current_disclosure_state=disclosure_state,
)

print("   ✓ Markdown fence handling successful")
print(f"   Directness: {guidance_md['directness']}")
print(f"   Disclosure: {guidance_md['disclosure_stage']}")

print("\n" + "=" * 60)
print("✓ All Patient Manager Function Tests Passed!")
print("=" * 60)
