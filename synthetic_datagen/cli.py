"""
CLI entrypoint for synthetic depression data generation.
"""
import argparse
import random
import json
import os
from typing import Optional
from dotenv import load_dotenv

from agents import set_default_openai_key, Runner, SQLiteSession, ModelSettings
from synthetic_datagen.generation.profile_generation import sample_patient_profile
from synthetic_datagen.generation.session_runner import (
    run_patient_doctor_session,
    build_patient_manager_agent,
)
from synthetic_datagen.config import (
    LOG_MINIMAL,
    LOG_LIGHT,
    LOG_HEAVY,
    LOG_LEVELS,
    DEFAULT_LOG_LEVEL,
    DEFAULT_NUM_SESSIONS,
)
from synthetic_datagen.generation.manager_logic import (
    build_patient_manager_input,
    parse_patient_manager_output,
)
from synthetic_datagen.prompts.patient_manager_prompt import PATIENT_MANAGER_SYSTEM_PROMPT
from synthetic_datagen.prompts.doctor_personas import DOCTOR_PERSONAS, sample_doctor_microstyle
from synthetic_datagen.utils.io import save_transcript
from synthetic_datagen import config


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic doctor-patient depression screening dialogues"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        choices=config.AVAILABLE_MODELS,
        help=f"OpenAI model to use (default: gpt-5-mini, options: {', '.join(config.AVAILABLE_MODELS)})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    parser.add_argument(
        "--num-sessions",
        type=int,
        default=DEFAULT_NUM_SESSIONS,
        help=f"Number of sessions to generate (default: {DEFAULT_NUM_SESSIONS})"
    )
    parser.add_argument(
        "--test-profile",
        action="store_true",
        help="Test mode: sample and print profile JSON without calling LLM"
    )
    parser.add_argument(
        "--test-patient-manager",
        action="store_true",
        help="Test mode: test patient manager with dummy input (requires API key)"
    )
    parser.add_argument(
        "--forced-template",
        type=str,
        default=None,
        help="Force specific Big Five template (e.g., NEUROTICISM_HIGH)"
    )
    parser.add_argument(
        "--forced-density",
        type=str,
        default=None,
        choices=["ULTRA_LOW", "LOW", "MED", "HIGH"],
        help="Force specific episode density level"
    )
    parser.add_argument(
        "--forced-persona",
        type=str,
        default=None,
        help="Force specific doctor persona (e.g., warm_validating)"
    )
    parser.add_argument(
        "--forced-age",
        type=str,
        default=None,
        help="Force specific age range (e.g., 16-19, 70-80)"
    )
    parser.add_argument(
        "--forced-trust",
        type=str,
        default=None,
        choices=["guarded", "neutral", "open"],
        help="Force specific trust level"
    )
    parser.add_argument(
        "--forced-verbosity",
        type=str,
        default=None,
        choices=["terse", "moderate", "detailed"],
        help="Force specific verbosity level"
    )
    parser.add_argument(
        "--forced-expressiveness",
        type=str,
        default=None,
        choices=["flat", "balanced", "intense"],
        help="Force specific expressiveness level"
    )
    parser.add_argument(
        "--forced-modifiers",
        type=str,
        default=None,
        help="Force specific modifiers (comma-separated, e.g., 'hostile,irritable')"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=LOG_LEVELS,
        help=f"Logging verbosity (default: {DEFAULT_LOG_LEVEL}). 'minimal' shows only dialogue, 'light' adds manager guidance, 'heavy' shows all parameters"
    )
    
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Set model in config
    config.OPENAI_MODEL = args.model
    print(f"Using model: {args.model}")
    
    # Set up random seed
    if args.seed is not None:
        seed = args.seed
        config.RANDOM_SEED = seed
    else:
        seed = random.randint(1, 1000000)
        config.RANDOM_SEED = seed
    
    print(f"Using random seed: {seed}")
    
    # Create RNG
    rng = random.Random(seed)
    
    # Build forced overrides dict
    forced_overrides = {}
    if args.forced_template:
        forced_overrides["template_id"] = args.forced_template
    if args.forced_density:
        forced_overrides["episode_density"] = args.forced_density
    if args.forced_age:
        forced_overrides["age_range"] = args.forced_age
    
    # Build forced voice style
    if args.forced_trust or args.forced_verbosity or args.forced_expressiveness:
        # Start with defaults or sample
        forced_voice_style = {}
        if args.forced_trust:
            forced_voice_style["trust"] = args.forced_trust
        if args.forced_verbosity:
            forced_voice_style["verbosity"] = args.forced_verbosity
        if args.forced_expressiveness:
            forced_voice_style["expressiveness"] = args.forced_expressiveness
        forced_overrides["voice_style_partial"] = forced_voice_style
    
    # Parse forced modifiers
    if args.forced_modifiers:
        forced_overrides["modifiers"] = [m.strip() for m in args.forced_modifiers.split(",")]
    
    # Test profile mode (Step 8: fast test path)
    if args.test_profile:
        print("\n=== TEST PROFILE MODE ===")
        print("Sampling patient profile without LLM calls...\n")
        
        profile = sample_patient_profile(rng, forced=forced_overrides if forced_overrides else None)
        
        # Print profile as JSON
        print(json.dumps(profile, indent=2, default=str))
        print("\nProfile sampled successfully. Exiting without LLM calls.")
        return
    
    # Read API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    set_default_openai_key(api_key)
    
    # P8: Test patient manager mode
    if args.test_patient_manager:
        print("\n=== TEST PATIENT MANAGER MODE ===")
        print("Testing patient manager with dummy scenario...\n")
        
        # Sample a patient profile
        profile = sample_patient_profile(rng, forced=forced_overrides if forced_overrides else None)
        template_id = profile["template_id"]
        template = profile["template"]
        personality = profile["personality"]
        depression_profile = profile["depression_profile"]
        
        print(f"Template: {template_id}")
        print(f"Voice style: {personality.get('VOICE_STYLE', {})}")
        print(f"Pacing: {personality.get('PACING', 'N/A')}")
        print(f"Episode density: {personality.get('EPISODE_DENSITY', 'N/A')}\n")
        
        # Create dummy context
        dummy_doctor_msg = "Over the past two weeks, how often have you felt down or hopeless?"
        dummy_last_move = "DSM"
        dummy_conversation = [
            {"role": "assistant", "content": "Hello, how are you doing today?"},
            {"role": "user", "content": "I'm okay, I guess."},
        ]
        
        # Initialize disclosure state
        trust = personality.get("VOICE_STYLE", {}).get("trust", "neutral")
        if trust == "guarded":
            disclosure_state = "MINIMIZE"
        elif trust == "open":
            disclosure_state = "OPEN"
        else:
            disclosure_state = "PARTIAL"
        
        print(f"Initial disclosure state: {disclosure_state}\n")
        
        # Build patient manager input
        patient_meta = {
            "template_id": template_id,
            "emphasized_symptoms": template.get("emphasized_symptoms", []),
            "modifiers": personality.get("MODIFIERS", []),
            "voice_style": personality.get("VOICE_STYLE", {}),
            "pacing": personality.get("PACING", "MED"),
            "episode_density": personality.get("EPISODE_DENSITY", "MED"),
        }
        
        pm_input = build_patient_manager_input(
            last_doctor_message=dummy_doctor_msg,
            last_doctor_move=dummy_last_move,
            conversation_history=dummy_conversation,
            patient_meta=patient_meta,
            depression_profile=depression_profile,
            disclosure_state=disclosure_state,
        )
        
        print("=== Patient Manager Input ===")
        print(pm_input)
        print("\n=== Calling Patient Manager Agent ===\n")
        
        # Create patient manager agent and call
        # Note: gpt-5-mini doesn't support temperature parameter
        session = SQLiteSession(":memory:")
        runner = Runner()
        if config.OPENAI_MODEL == "gpt-5-mini":
            model_settings = ModelSettings(max_tokens=300)
        else:
            model_settings = ModelSettings(temperature=0.5, max_tokens=300)
        patient_manager_agent = build_patient_manager_agent(
            model_settings=model_settings
        )
        
        try:
            pm_res = runner.run_sync(patient_manager_agent, pm_input, session=session)
            pm_output = pm_res.final_output
            
            print("=== Patient Manager Raw Output ===")
            print(pm_output)
            print()
            
            # Parse output
            guidance = parse_patient_manager_output(
                pm_output,
                base_voice_style=personality.get("VOICE_STYLE", {}),
                current_disclosure_state=disclosure_state,
            )
            
            print("=== Parsed Guidance ===")
            print(json.dumps(guidance, indent=2))
            print("\n✓ Patient manager test completed successfully.")
            
        except Exception as e:
            print(f"\n✗ Error testing patient manager: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    print(f"\n=== GENERATING {args.num_sessions} SESSION(S) ===\n")
    
    # Generate sessions
    for i in range(args.num_sessions):
        print(f"\n--- Session {i + 1}/{args.num_sessions} ---")
        
        # Build forced agent profile if needed
        forced_agent_profile = None
        if forced_overrides or args.forced_persona:
            # Sample a profile first (this calls the background writer)
            profile = sample_patient_profile(rng, forced=forced_overrides if forced_overrides else None)
            forced_agent_profile = {
                "template_id": profile["template_id"],
                "template": profile["template"],
                "personality": profile["personality"],
                "depression_profile": profile["depression_profile"],
                "life_background": profile.get("life_background"),  # Include life background!
                "background_writer_prompt_trace": profile.get("background_writer_prompt_trace"),
            }
            # Select doctor persona (forced or random)
            if args.forced_persona:
                forced_agent_profile["doctor_persona_id"] = args.forced_persona
            else:
                # Randomly select persona using the RNG
                forced_agent_profile["doctor_persona_id"] = rng.choice(DOCTOR_PERSONAS)["id"]
            # Sample microstyle
            forced_agent_profile["doctor_microstyle"] = sample_doctor_microstyle()
        
        # Map log level string to constant
        log_level_map = {
            "minimal": LOG_MINIMAL,
            "light": LOG_LIGHT,
            "heavy": LOG_HEAVY,
        }
        log_level = log_level_map.get(args.log_level, DEFAULT_LOG_LEVEL)
        
        # Run session
        try:
            session_data = run_patient_doctor_session(
                rng=rng,
                use_random_agent=(forced_agent_profile is None),
                forced_agent_profile=forced_agent_profile,
                log_level=log_level,
            )
            
            # Save transcript
            filepath = save_transcript(session_data)
            print(f"\n✓ Session saved: {filepath}")
            print(f"  Template: {session_data['template_id']}")
            print(f"  Persona: {session_data['persona_id']}")
            print(f"  Agent ID: {session_data['agent_id']}")
            
        except Exception as e:
            print(f"\n✗ Error generating session {i + 1}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== COMPLETE: Generated {args.num_sessions} session(s) ===")


if __name__ == "__main__":
    main()
