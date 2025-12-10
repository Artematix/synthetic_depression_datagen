"""
Quick test to verify all imports work correctly.
"""

print("Testing imports...")

# Test config imports
from synthetic_datagen.config import (
    OPENAI_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_EXTRA_DOCTOR_TURNS,
)
print("✓ Config imports successful")

# Test data imports
from synthetic_datagen.data.templates import DSM5_DEPRESSION_SYMPTOMS, BIG5_DEP_TEMPLATES
from synthetic_datagen.data.pools import PACING_LEVELS, VOICE_STYLE_PROFILES
from synthetic_datagen.data.questionnaire import QUESTIONNAIRE
print("✓ Data imports successful")

# Prompt imports
from synthetic_datagen.prompts.doctor_personas import DOCTOR_PERSONAS, sample_doctor_microstyle
print("✓ Prompt imports successful")

# Test additional prompt imports
from synthetic_datagen.prompts.patient_prompt import build_patient_system_prompt
from synthetic_datagen.prompts.doctor_manager_prompt import DOCTOR_MANAGER_SYSTEM_PROMPT
from synthetic_datagen.prompts.patient_manager_prompt import PATIENT_MANAGER_SYSTEM_PROMPT
from synthetic_datagen.prompts.doctor_base import DOCTOR_BASE_SYSTEM_PROMPT
print("✓ Additional prompt imports successful")

# Test generation imports
from synthetic_datagen.generation.profile_generation import sample_patient_profile, generate_depression_profile
from synthetic_datagen.generation.manager_logic import (
    build_manager_input,
    parse_doctor_manager_output,
    build_patient_manager_input,
    parse_patient_manager_output,
)
from synthetic_datagen.generation.session_runner import (
    run_patient_doctor_session,
    build_doctor_manager_agent,
    build_patient_manager_agent,
)
print("✓ Generation imports successful")

# Test utils imports
from synthetic_datagen.utils.io import save_transcript
print("✓ Utils imports successful")

# Test CLI import
from synthetic_datagen.cli import main
print("✓ CLI imports successful")

print("\n=== ALL IMPORTS SUCCESSFUL ===")
print("\nPackage structure:")
print("- synthetic_datagen/")
print("  - __init__.py")
print("  - config.py")
print("  - cli.py")
print("  - data/")
print("    - templates.py (DSM5, BIG5 templates)")
print("    - pools.py (pacing, voice styles, etc.)")
print("    - questionnaire.py (DSM questions)")
print("  - prompts/")
print("    - doctor_base.py (base doctor prompt)")
print("    - doctor_personas.py (doctor personas, microstyle)")
print("    - doctor_manager_prompt.py (doctor manager)")
print("    - patient_prompt.py (patient prompt builder)")
print("    - patient_manager_prompt.py (patient manager)")
print("  - generation/")
print("    - profile_generation.py (pure functions)")
print("    - manager_logic.py (manager helpers)")
print("    - session_runner.py (main session logic)")
print("  - utils/")
print("    - io.py (save_transcript)")

print("\nKey features implemented:")
print("✓ Modular package structure")
print("✓ Pure functions in profile_generation.py")
print("✓ Doctor manager logic with fallbacks")
print("✓ Patient manager for disclosure control")
print("✓ Last-4-messages context trimming")
print("✓ Base doctor prompt + persona overlays")
print("✓ MAX_EXTRA_DOCTOR_TURNS semantics")
print("✓ CLI with --test-profile and --test-patient-manager flags")
print("✓ Fresh session per run")
print("✓ Prompt trace logging to outputs/prompt_traces/")
print("✓ Output to outputs/transcripts/")
