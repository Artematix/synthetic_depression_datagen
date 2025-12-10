"""
Main session runner for doctor-patient dialogue generation.
"""
import os
import json
import random
import hashlib
import datetime
from typing import Dict, Any, Optional, List

from agents import Agent, ModelSettings, Runner, SQLiteSession
from synthetic_datagen.config import (
    OPENAI_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    RANDOM_TEMPERATURE,
    RANDOM_MAX_TOKENS,
    PACING_TARGETS,
    EXTENDED_PACING_PERSONAS,
    NUM_DSM_ITEMS,
    MANAGER_TEMPERATURE,
    MANAGER_MAX_TOKENS,
    DEFAULT_DOCTOR_PERSONA_ID,
    BUFFER_TURNS,
    LOG_MINIMAL,
    LOG_LIGHT,
    LOG_HEAVY,
)


def get_model_settings(max_tokens: int = 1000) -> ModelSettings:
    """
    Get ModelSettings appropriate for the current model.
    
    Note: gpt-5-mini does not support the temperature parameter.
    For gpt-5-mini, we return None to omit model_settings entirely.
    """
    if OPENAI_MODEL == "gpt-5-mini":
        # gpt-5-mini doesn't support temperature parameter - return None to use defaults
        return None
    else:
        # Other models support temperature
        return ModelSettings(temperature=DEFAULT_TEMPERATURE, max_tokens=max_tokens)
from synthetic_datagen.data.templates import BIG5_DEP_TEMPLATES
from synthetic_datagen.data.questionnaire import DSM_ITEMS
from synthetic_datagen.prompts.doctor_personas import (
    DOCTOR_PERSONAS,
    sample_doctor_microstyle,
    build_doctor_system_prompt,
)
from synthetic_datagen.generation.profile_generation import sample_patient_profile
from synthetic_datagen.generation.manager_logic import (
    build_manager_input,
    parse_doctor_manager_output,
    build_patient_manager_input,
    parse_patient_manager_output,
)
from synthetic_datagen.prompts.patient_prompt import build_patient_system_prompt
from synthetic_datagen.prompts.doctor_manager_prompt import (
    DOCTOR_MANAGER_SYSTEM_PROMPT,
    DOCTOR_MANAGER_LOW_TURNS_SYSTEM_PROMPT,
    DOCTOR_MANAGER_FORCE_DSM_SYSTEM_PROMPT,
)
from synthetic_datagen.prompts.patient_manager_prompt import PATIENT_MANAGER_SYSTEM_PROMPT


# ============================================================================
# LOGGING HELPERS
# ============================================================================

def print_banner(text: str, char: str = "=", width: int = 80):
    """Print a banner with text centered."""
    print()
    print(char * width)
    padding = (width - len(text) - 2) // 2
    print(f"{char}{' ' * padding}{text}{' ' * (width - padding - len(text) - 2)}{char}")
    print(char * width)
    print()


def print_section(title: str, width: int = 80):
    """Print a section header."""
    print()
    print("-" * width)
    print(f"  {title}")
    print("-" * width)


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n  >>> {title}")


def print_session_header(
    doctor_persona_id: str,
    doctor_microstyle: Dict[str, Any],
    template_id: str,
    personality: Dict[str, Any],
    life_background: Any,
    depression_profile: Dict[str, str],
    template: Dict[str, Any],
    target_turns_per_symptom: float,
    max_doctor_turns: int,
    log_level: str,
):
    """Print comprehensive session header with all details."""
    # Big separator for new session
    print("\n" * 3)
    print("=" * 80)
    print("=" * 80)
    print_banner("NEW SESSION", "=", 80)
    print("=" * 80)
    print("=" * 80)
    
    # Doctor details
    print_section("DOCTOR PROFILE")
    print(f"  Persona ID:     {doctor_persona_id}")
    print(f"  Warmth:         {doctor_microstyle.get('warmth', 'N/A')}")
    print(f"  Directness:     {doctor_microstyle.get('directness', 'N/A')}")
    print(f"  Pacing:         {doctor_microstyle.get('pacing', 'N/A')}")
    print(f"  Humor:          {doctor_microstyle.get('humor', 'N/A')}")
    print(f"  Animation:      {doctor_microstyle.get('animation', 'N/A')}")
    
    # Patient details
    print_section("PATIENT PROFILE")
    print(f"  Template ID:    {template_id}")
    print(f"  Big Five:       {personality.get('BIG5_TEMPLATE', 'N/A')}")
    print(f"  Specifier:      {personality.get('SPECIFIER_HINT', 'N/A')}")
    
    vs = personality.get("VOICE_STYLE", {})
    print(f"\n  Voice Style:")
    print(f"    Verbosity:      {vs.get('verbosity', 'N/A')}")
    print(f"    Expressiveness: {vs.get('expressiveness', 'N/A')}")
    print(f"    Trust:          {vs.get('trust', 'N/A')}")
    print(f"    Intellect:      {vs.get('intellect', 'N/A')}")
    print(f"    Humor:          {vs.get('humor', 'N/A')}")
    
    print(f"\n  Behavior:")
    print(f"    Pacing:         {personality.get('PACING', 'N/A')}")
    print(f"    Episode Density:{personality.get('EPISODE_DENSITY', 'N/A')}")
    print(f"    Modifiers:      {personality.get('MODIFIERS', [])}")
    print(f"    Context Domains:{personality.get('CONTEXT_DOMAINS', [])}")
    
    # Personal background
    personal_bg = personality.get("PERSONAL_BACKGROUND", {})
    if personal_bg:
        print(f"\n  Personal Background:")
        for key, val in personal_bg.items():
            print(f"    {key}: {val}")
    
    # Life background (rich detail)
    if life_background:
        print_section("LIFE BACKGROUND")
        print(f"  Name:           {life_background.name}")
        print(f"  Age Range:      {life_background.age_range}")
        print(f"  Pronouns:       {life_background.pronouns}")
        print(f"  Core Roles:     {', '.join(life_background.core_roles) if life_background.core_roles else 'N/A'}")
        
        if life_background.core_relationships:
            print(f"\n  Key Relationships:")
            for rel in life_background.core_relationships:
                print(f"    • {rel}")
        
        print(f"\n  Core Stressors:")
        print(f"    {life_background.core_stressor_summary}")
        
        # Life facets by salience
        high_sal = [f for f in life_background.life_facets if f.salience == "high"]
        med_sal = [f for f in life_background.life_facets if f.salience == "med"]
        
        if high_sal:
            print(f"\n  High-Salience Life Facets:")
            for facet in high_sal:
                print(f"    [{facet.category}]")
                # Wrap long descriptions
                desc = facet.description
                while len(desc) > 70:
                    print(f"      {desc[:70]}")
                    desc = desc[70:]
                print(f"      {desc}")
        
        if med_sal:
            print(f"\n  Medium-Salience Life Facets:")
            for facet in med_sal:
                print(f"    [{facet.category}]")
                desc = facet.description
                while len(desc) > 70:
                    print(f"      {desc[:70]}")
                    desc = desc[70:]
                print(f"      {desc}")
    
    # Depression profile
    print_section("DEPRESSION PROFILE (Ground Truth)")
    print(f"  Emphasized Symptoms: {template.get('emphasized_symptoms', [])}")
    print()
    for symptom, freq in depression_profile.items():
        indicator = "██" if freq in ["SOME", "OFTEN"] else "░░" if freq == "RARE" else "  "
        print(f"  {indicator} {symptom}: {freq}")
    
    # Pacing info
    print_section("PACING CONFIGURATION")
    print(f"  Target turns/symptom: {target_turns_per_symptom}")
    print(f"  Max doctor turns:     {max_doctor_turns}")
    print(f"  DSM items to cover:   {NUM_DSM_ITEMS}")
    
    # Start interview section
    print()
    print("=" * 80)
    print_banner("INTERVIEW BEGIN", "=", 80)
    print("=" * 80)


def print_doctor_turn_header(turn_num: int, mode: str, ratio: float = None, target: float = None, log_level: str = LOG_LIGHT):
    """Print doctor turn header based on log level."""
    if log_level == LOG_MINIMAL:
        print(f"\n--- Turn {turn_num} ---")
    elif log_level == LOG_LIGHT:
        if ratio is not None:
            print(f"\n--- Doctor Turn {turn_num} [{mode.upper()}] (ratio={ratio:.2f}, target={target}) ---")
        else:
            print(f"\n--- Doctor Turn {turn_num} [{mode.upper()}] ---")
    else:  # HEAVY
        print()
        print("=" * 80)
        if ratio is not None:
            print(f"  DOCTOR TURN {turn_num} | Mode: {mode.upper()} | Ratio: {ratio:.2f} | Target: {target}")
        else:
            print(f"  DOCTOR TURN {turn_num} | Mode: {mode.upper()}")
        print("=" * 80)


def print_doctor_manager_decision(decision: Dict[str, Any], log_level: str):
    """Print doctor manager decision based on log level."""
    if log_level == LOG_MINIMAL:
        return  # No manager output in minimal
    
    next_action = decision.get("next_action", "N/A")
    reason = decision.get("reason", "N/A")
    instruction = decision.get("doctor_instruction", "N/A")
    dsm_key = decision.get("dsm_symptom_key", "")
    
    if log_level == LOG_LIGHT:
        print(f"  Manager: {next_action}" + (f" → {dsm_key}" if dsm_key else ""))
        print(f"  Reason: {reason}")
        print(f"  Instruction: {instruction[:100]}..." if len(instruction) > 100 else f"  Instruction: {instruction}")
    else:  # HEAVY
        print_subsection("DOCTOR MANAGER DECISION")
        print(f"    Next Action:    {next_action}")
        if dsm_key:
            print(f"    DSM Symptom:    {dsm_key}")
        print(f"    Reason:         {reason}")
        print(f"    Instruction:")
        # Wrap instruction
        inst = instruction
        while len(inst) > 70:
            print(f"      {inst[:70]}")
            inst = inst[70:]
        print(f"      {inst}")


def print_patient_turn_header(turn_num: int, log_level: str):
    """Print patient turn header based on log level."""
    if log_level == LOG_MINIMAL:
        return  # Handled in main output
    elif log_level == LOG_LIGHT:
        print(f"\n--- Patient Turn {turn_num} ---")
    else:  # HEAVY
        print()
        print("-" * 80)
        print(f"  PATIENT TURN {turn_num}")
        print("-" * 80)


def print_patient_manager_guidance(guidance: Dict[str, Any], ground_truth_note: str, log_level: str):
    """Print patient manager guidance based on log level."""
    if log_level == LOG_MINIMAL:
        return  # No manager output in minimal
    
    if log_level == LOG_LIGHT:
        print(f"  Disclosure: {guidance.get('disclosure_stage', 'N/A')} | Length: {guidance.get('target_length', 'N/A')}")
        print(f"  Instruction: {guidance.get('patient_instruction', 'N/A')[:100]}..." if len(guidance.get('patient_instruction', '')) > 100 else f"  Instruction: {guidance.get('patient_instruction', 'N/A')}")
        if ground_truth_note:
            print(f"  {ground_truth_note}")
    else:  # HEAVY
        print_subsection("PATIENT MANAGER GUIDANCE")
        print(f"    Directness:       {guidance.get('directness', 'N/A')}")
        print(f"    Disclosure Stage: {guidance.get('disclosure_stage', 'N/A')}")
        print(f"    Target Length:    {guidance.get('target_length', 'N/A')}")
        print(f"    Emotional State:  {guidance.get('emotional_state', 'N/A')}")
        print(f"    Tone Tags:        {guidance.get('tone_tags', [])}")
        
        if guidance.get('key_points_to_reveal'):
            print(f"    Key Points to Reveal:")
            for point in guidance.get('key_points_to_reveal', []):
                print(f"      • {point}")
        
        if guidance.get('key_points_to_avoid'):
            print(f"    Key Points to Avoid:")
            for point in guidance.get('key_points_to_avoid', []):
                print(f"      • {point}")
        
        print(f"    Instruction:")
        inst = guidance.get('patient_instruction', 'N/A')
        while len(inst) > 70:
            print(f"      {inst[:70]}")
            inst = inst[70:]
        print(f"      {inst}")
        
        if ground_truth_note:
            print(f"\n    {ground_truth_note}")


def print_dialogue_line(speaker: str, text: str, log_level: str):
    """Print a dialogue line based on log level."""
    if log_level == LOG_MINIMAL:
        print(f"\n{speaker.upper()}: {text}")
    elif log_level == LOG_LIGHT:
        print(f"\n  {speaker.capitalize()}: {text}")
    else:  # HEAVY
        print_subsection(f"{speaker.upper()} SAYS")
        # Wrap text
        remaining = text
        while len(remaining) > 70:
            print(f"    {remaining[:70]}")
            remaining = remaining[70:]
        print(f"    {remaining}")


def print_token_summary(token_usage: Dict[str, Dict[str, int]], log_level: str):
    """Print token usage summary."""
    if log_level == LOG_MINIMAL:
        return  # No token summary in minimal
    
    print()
    print("=" * 80)
    print_banner("TOKEN USAGE SUMMARY", "=", 80)
    print("=" * 80)
    
    total_tokens = sum(agent["total_tokens"] for agent in token_usage.values())
    total_input = sum(agent["input_tokens"] for agent in token_usage.values())
    total_output = sum(agent["output_tokens"] for agent in token_usage.values())
    
    for agent_name, usage in token_usage.items():
        print(f"  {agent_name}:")
        print(f"    Input:  {usage['input_tokens']:,} tokens")
        print(f"    Output: {usage['output_tokens']:,} tokens")
        print(f"    Total:  {usage['total_tokens']:,} tokens")
    
    print(f"\n  GRAND TOTAL:")
    print(f"    Input:  {total_input:,} tokens")
    print(f"    Output: {total_output:,} tokens")
    print(f"    Total:  {total_tokens:,} tokens")


# ============================================================================
# AGENT HELPERS
# ============================================================================

def init_disclosure_state(personality: Dict[str, Any]) -> str:
    """
    Initialize disclosure state based on patient profile.
    
    Parameters
    ----------
    personality : dict
        Patient personality dict with VOICE_STYLE
        
    Returns
    -------
    str
        Initial disclosure state: "MINIMIZE", "PARTIAL", or "OPEN"
    """
    voice_style = personality.get("VOICE_STYLE", {})
    trust = voice_style.get("trust", "neutral")
    verbosity = voice_style.get("verbosity", "moderate")
    
    # Guarded or terse -> MINIMIZE
    if trust == "guarded" or verbosity == "terse":
        return "MINIMIZE"
    # Open and moderate/detailed -> OPEN
    elif trust == "open" and verbosity in ["moderate", "detailed"]:
        return "OPEN"
    # Neutral or other combinations -> PARTIAL
    else:
        return "PARTIAL"


def build_doctor_manager_agent(system_prompt: str, model_settings: ModelSettings = None) -> Agent:
    """
    Return a stateful DoctorConversationManager agent.
    This agent decides doctor's next action: FOLLOW_UP, RAPPORT, or DSM.
    
    Parameters
    ----------
    system_prompt : str
        The system prompt to use (normal or low-turns mode)
    model_settings : ModelSettings, optional
        Model settings for the agent (None for gpt-5-mini)
    """
    if model_settings is not None:
        return Agent(
            name="DoctorConversationManager",
            model=OPENAI_MODEL,
            instructions=system_prompt,
            model_settings=model_settings,
        )
    else:
        return Agent(
            name="DoctorConversationManager",
            model=OPENAI_MODEL,
            instructions=system_prompt,
        )


def build_doctor_agent(*, instructions: str, model_settings: ModelSettings = None) -> Agent:
    """Return a stateful doctor agent with its persona-specific system prompt."""
    if model_settings is not None:
        return Agent(
            name="SimDoctor",
            model=OPENAI_MODEL,
            instructions=instructions,
            model_settings=model_settings,
        )
    else:
        return Agent(
            name="SimDoctor",
            model=OPENAI_MODEL,
            instructions=instructions,
        )


def build_patient_agent(*, patient_system_prompt: str, model_settings: ModelSettings = None) -> Agent:
    """Return a stateful patient agent with its personality prompt."""
    if model_settings is not None:
        return Agent(
            name="SimPatient",
            model=OPENAI_MODEL,
            instructions=patient_system_prompt,
            model_settings=model_settings,
        )
    else:
        return Agent(
            name="SimPatient",
            model=OPENAI_MODEL,
            instructions=patient_system_prompt,
        )


def build_patient_manager_agent(model_settings: ModelSettings = None) -> Agent:
    """
    Return a stateful PatientConversationManager agent.
    This agent decides how the patient should respond (directness, disclosure, tone).
    """
    if model_settings is not None:
        return Agent(
            name="PatientConversationManager",
            model=OPENAI_MODEL,
            instructions=PATIENT_MANAGER_SYSTEM_PROMPT,
            model_settings=model_settings,
        )
    else:
        return Agent(
            name="PatientConversationManager",
            model=OPENAI_MODEL,
            instructions=PATIENT_MANAGER_SYSTEM_PROMPT,
        )


# ============================================================================
# MAIN SESSION RUNNER
# ============================================================================

def run_patient_doctor_session(
    *,
    rng: random.Random,
    use_random_agent: bool = False,
    forced_agent_profile: Optional[Dict[str, Any]] = None,
    log_level: str = LOG_LIGHT,
) -> Dict[str, Any]:
    """
    Execute a multi-turn doctor-patient dialogue with stateful agents.
    
    Parameters
    ----------
    rng : random.Random
        Random number generator for reproducibility
    use_random_agent : bool
        Whether to sample a random patient profile
    forced_agent_profile : dict, optional
        Forced profile overrides for testing
    log_level : str
        Logging verbosity: "minimal", "light", or "heavy"
        
    Returns
    -------
    dict
        Session data with conversation, metadata, and ground truth
    """
    # Step 3: Create sessions for conversational agents (doctor and patient are stateful)
    doctor_session = SQLiteSession(":memory:")
    patient_session = SQLiteSession(":memory:")
    runner = Runner()
    
    # Token usage tracking per agent
    token_usage = {
        "doctor": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "patient": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "doctor_manager": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "patient_manager": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    
    # Agent Profile Generation
    background_writer_prompt_trace = None  # Track background writer trace
    if use_random_agent or forced_agent_profile is None:
        # Sample patient profile using the RNG
        profile = sample_patient_profile(rng)
        template_id = profile["template_id"]
        template = profile["template"]
        personality = profile["personality"]
        depression_profile = profile["depression_profile"]
        life_background = profile.get("life_background")
        background_writer_prompt_trace = profile.get("background_writer_prompt_trace")
        
        # Select doctor persona
        doctor_persona = rng.choice(DOCTOR_PERSONAS)
        doctor_persona_id = doctor_persona["id"]
        
        # Sample per-session doctor microstyle
        doctor_microstyle = sample_doctor_microstyle()
    else:
        # Use forced profile
        template_id = forced_agent_profile["template_id"]
        template = BIG5_DEP_TEMPLATES[template_id]
        personality = forced_agent_profile["personality"]
        depression_profile = forced_agent_profile["depression_profile"]
        life_background = forced_agent_profile.get("life_background")
        background_writer_prompt_trace = forced_agent_profile.get("background_writer_prompt_trace")
        doctor_persona_id = forced_agent_profile.get("doctor_persona_id", DEFAULT_DOCTOR_PERSONA_ID)
        doctor_persona = next((p for p in DOCTOR_PERSONAS if p["id"] == doctor_persona_id), DOCTOR_PERSONAS[0])
        doctor_microstyle = forced_agent_profile.get("doctor_microstyle", sample_doctor_microstyle())
    
    # Build patient system prompt
    patient_system_prompt = build_patient_system_prompt(
        personality=personality,
        depression_profile=depression_profile,
        life_background=life_background,
    )
    
    # Deterministic agent_id - includes all major generation levers
    # This ensures unique IDs when any patient OR doctor parameter changes
    voice_style = personality.get("VOICE_STYLE", {})
    modifiers = personality.get("MODIFIERS", [])
    profile_key_parts = [
        # Patient levers
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
        # Doctor levers
        f"PERSONA={doctor_persona_id}",
        f"D_WARMTH={doctor_microstyle.get('warmth', 'med')}",
        f"D_DIRECT={doctor_microstyle.get('directness', 'med')}",
        f"D_PACING={doctor_microstyle.get('pacing', 'med')}",
        f"D_HUMOR={doctor_microstyle.get('humor', 'none')}",
        f"D_ANIMATION={doctor_microstyle.get('animation', 'med')}",
    ]
    # Add depression profile (symptom frequencies)
    for symptom in sorted(depression_profile.keys()):
        profile_key_parts.append(f"{symptom}={depression_profile[symptom]}")
    
    profile_key = "|".join(profile_key_parts)
    agent_hash = hashlib.sha256(profile_key.encode()).hexdigest()[:16]
    agent_id = f"AGENT_{agent_hash}"
    
    # Session-level generation settings
    patient_temperature = (
        round(rng.uniform(0.6, 1.4), 2)
        if RANDOM_TEMPERATURE else DEFAULT_TEMPERATURE
    )
    
    doctor_temperature = (
        round(rng.uniform(0.6, 1.4), 2)
        if RANDOM_TEMPERATURE else DEFAULT_TEMPERATURE
    )
    
    max_tokens = (
        rng.randint(200, 400)
        if RANDOM_MAX_TOKENS else DEFAULT_MAX_TOKENS
    )
    
    # Build doctor system prompt with base + persona + microstyle + life_background
    doctor_instructions = build_doctor_system_prompt(
        doctor_persona["system_prompt"],
        doctor_microstyle,
        life_background
    )
    
    # Instantiate agents
    doctor_agent = build_doctor_agent(
        instructions=doctor_instructions,
        model_settings=get_model_settings(max_tokens)
    )
    
    patient_agent = build_patient_agent(
        patient_system_prompt=patient_system_prompt,
        model_settings=get_model_settings(max_tokens)
    )
    
    # Note: doctor_manager_agent will be rebuilt per-turn with appropriate system prompt
    # This initial one is not used - we build fresh agents in the loop
    
    # Instantiate patient manager agent
    patient_manager_agent = build_patient_manager_agent(
        model_settings=get_model_settings(300)
    )
    
    # Initialize disclosure_state using helper function
    disclosure_state = init_disclosure_state(personality)
    
    # Compute pacing-based turn budget
    microstyle_pacing = doctor_microstyle.get("pacing", "med")
    base_target = PACING_TARGETS.get(microstyle_pacing, 2.0)
    
    # Adjust for extended pacing personas
    if doctor_persona_id in EXTENDED_PACING_PERSONAS:
        target_turns_per_symptom = min(base_target + 0.5, 3.0)
    else:
        target_turns_per_symptom = base_target
    
    # Compute turn budgets (excluding greeting)
    import math
    base_turn_budget = math.ceil(target_turns_per_symptom * NUM_DSM_ITEMS)
    max_doctor_turns = base_turn_budget + BUFFER_TURNS + 1  # Total allowed turns
    # dsm_turn_budget is used for ratio calculation (doesn't include buffer)
    dsm_turn_budget = base_turn_budget  # Turns allocated for DSM coverage
    
    # Generate patient background summary and risk summary
    personal_bg = personality.get("PERSONAL_BACKGROUND", {})
    bg_parts = []
    for k, v in personal_bg.items():
        bg_parts.append(v)
    patient_background_summary = ", ".join(bg_parts) if bg_parts else "no specific background provided"
    
    # Generate risk summary from depression profile
    suicidal_freq = depression_profile.get("Recurrent thoughts of death or suicide", "NONE")
    depressed_freq = depression_profile.get("Depressed mood", "NONE")
    
    if suicidal_freq in ["SOME", "OFTEN"]:
        risk_summary = "high risk: recent suicidal thoughts"
    elif depressed_freq in ["OFTEN"] or suicidal_freq == "RARE":
        risk_summary = "moderate depression, monitor closely"
    elif depressed_freq in ["SOME"]:
        risk_summary = "mild to moderate depression, no immediate risk"
    else:
        risk_summary = "low risk, minimal symptoms"
    
    # Print session header
    print_session_header(
        doctor_persona_id=doctor_persona_id,
        doctor_microstyle=doctor_microstyle,
        template_id=template_id,
        personality=personality,
        life_background=life_background,
        depression_profile=depression_profile,
        template=template,
        target_turns_per_symptom=target_turns_per_symptom,
        max_doctor_turns=max_doctor_turns,
        log_level=log_level,
    )
    
    # Dialogue loop initialization
    conversation_history: List[Dict[str, str]] = []
    dsm_pool = list(DSM_ITEMS)  # Copy of DSM symptom keys
    asked_question_order: List[str] = []
    patient_manager_decisions: List[Dict[str, Any]] = []  # Track patient manager decisions
    doctor_manager_decisions: List[Dict[str, Any]] = []  # Track doctor manager decisions
    prompt_traces: List[Dict[str, Any]] = []  # Track all prompts and outputs
    
    # Add background writer prompt trace if available (captured during profile generation)
    if background_writer_prompt_trace:
        prompt_traces.append(background_writer_prompt_trace)
    
    # Doctor turns elapsed counts extra turns beyond greeting
    doctor_turns_elapsed = 0
    
    # Doctor's first greeting (not counted in extra turns)
    doctor_first_greeting = doctor_persona["first_greeting"]
    
    # Print doctor greeting with consistent formatting
    if log_level != LOG_MINIMAL:
        print()
        print("=" * 60)
        print(f"  DOCTOR TURN 0 (GREETING)")
        print("=" * 60)
    print_dialogue_line("doctor", doctor_first_greeting, log_level)
    
    conversation_history.append({"role": "assistant", "content": doctor_first_greeting})
    
    # Patient's first response - route through patient manager
    if log_level != LOG_MINIMAL:
        print()
        print("-" * 60)
        print(f"  PATIENT TURN 1")
        print("-" * 60)
    
    # Build patient manager input for first response
    patient_manager_meta = {
        "template_id": template_id,
        "emphasized_symptoms": template.get("emphasized_symptoms", []),
        "modifiers": personality.get("MODIFIERS", []),
        "voice_style": personality.get("VOICE_STYLE", {}),
        "pacing": personality.get("PACING", "MED"),
        "episode_density": personality.get("EPISODE_DENSITY", "MED"),
    }
    
    pm_input_first = build_patient_manager_input(
        last_doctor_message=doctor_first_greeting,
        last_doctor_move="RAPPORT",  # Greeting is treated as RAPPORT
        conversation_history=conversation_history,
        patient_meta=patient_manager_meta,
        depression_profile=depression_profile,
        disclosure_state=disclosure_state,
    )
    
    # Call patient manager with fresh session (stateless)
    pm_session_first = SQLiteSession(":memory:")
    pm_res_first = runner.run_sync(patient_manager_agent, pm_input_first, session=pm_session_first)
    pm_output_first = pm_res_first.final_output
    
    # Track token usage
    if hasattr(pm_res_first, 'context_wrapper') and hasattr(pm_res_first.context_wrapper, 'usage'):
        usage = pm_res_first.context_wrapper.usage
        token_usage["patient_manager"]["input_tokens"] += usage.input_tokens
        token_usage["patient_manager"]["output_tokens"] += usage.output_tokens
        token_usage["patient_manager"]["total_tokens"] += usage.total_tokens
    
    # Log prompt trace
    prompt_traces.append({
        "agent": "patient_manager",
        "turn_index": 1,
        "system_prompt": PATIENT_MANAGER_SYSTEM_PROMPT,
        "input": pm_input_first,
        "output": pm_output_first,
    })
    
    # Parse manager guidance
    guidance_first = parse_patient_manager_output(
        pm_output_first,
        base_voice_style=personality.get("VOICE_STYLE", {}),
        current_disclosure_state=disclosure_state,
    )
    
    if log_level != LOG_MINIMAL:
        print("\n  [MANAGER GUIDANCE]")
        print(f"    Disclosure: {guidance_first['disclosure_stage']} | Length: {guidance_first['target_length']}")
        print(f"    Emotional: {guidance_first.get('emotional_state', 'neutral')}")
        print(f"    Instruction: {guidance_first['patient_instruction']}")
    
    # Update disclosure_state
    disclosure_state = guidance_first["disclosure_stage"]
    
    # Store first guidance
    patient_manager_decisions.append({
        "turn": 1,
        "doctor_move": "RAPPORT",
        "guidance": guidance_first,
    })
    
    # Build input to patient agent with guidance
    patient_input_first = f"""PATIENT_GUIDANCE:
directness: {guidance_first['directness']}
disclosure_stage: {guidance_first['disclosure_stage']}
target_length: {guidance_first['target_length']}
emotional_state: {guidance_first.get('emotional_state', 'neutral')}
tone_tags: {guidance_first['tone_tags']}
key_points_to_reveal: {guidance_first['key_points_to_reveal']}
key_points_to_avoid: {guidance_first['key_points_to_avoid']}
instruction: {guidance_first['patient_instruction']}

DOCTOR_MESSAGE:
{doctor_first_greeting}
"""
    
    # Patient's turn with guidance
    res = runner.run_sync(
        patient_agent,
        patient_input_first,
        session=patient_session,
    )
    patient_reply = res.final_output
    
    # Track token usage
    if hasattr(res, 'context_wrapper') and hasattr(res.context_wrapper, 'usage'):
        usage = res.context_wrapper.usage
        token_usage["patient"]["input_tokens"] += usage.input_tokens
        token_usage["patient"]["output_tokens"] += usage.output_tokens
        token_usage["patient"]["total_tokens"] += usage.total_tokens
    
    # Log prompt trace
    prompt_traces.append({
        "agent": "patient",
        "turn_index": 1,
        "system_prompt": patient_system_prompt,
        "input": patient_input_first,
        "output": patient_reply,
    })
    
    print_dialogue_line("patient", patient_reply, log_level)
    conversation_history.append({"role": "user", "content": patient_reply})
    
    # Print pacing information
    if log_level != LOG_MINIMAL:
        print(f"\n{'=' * 60}")
        print(f"PACING CONFIGURATION:")
        print(f"  Target turns per symptom: {target_turns_per_symptom}")
        print(f"  Max doctor turns (excl. greeting): {max_doctor_turns}")
        print(f"  DSM items to cover: {NUM_DSM_ITEMS}")
        print(f"{'=' * 60}\n")
    
    # Main dialogue loop (extra turns beyond greeting)
    while doctor_turns_elapsed < max_doctor_turns:
        # Calculate remaining resources
        # remaining_total_turns: actual turns left (used for force-DSM guard)
        # remaining_dsm_turns: turns left for ratio calculation (doesn't count buffer)
        remaining_total_turns = max_doctor_turns - doctor_turns_elapsed
        remaining_dsm_turns = max(0, dsm_turn_budget - doctor_turns_elapsed)
        remaining_dsm_items = len(dsm_pool)
        
        # Check if DSM screening is complete
        if remaining_dsm_items == 0:
            # Post-DSM phase: use post-DSM manager
            if log_level != LOG_MINIMAL:
                print()
                print("=" * 60)
                print(f"  DOCTOR TURN {doctor_turns_elapsed + 1} (POST-DSM)")
                print("=" * 60)
                print("\n  [MANAGER GUIDANCE]")
            
            # Build patient metadata for post-DSM manager
            vs = personality.get("VOICE_STYLE", {})
            patient_meta = {
                "template_id": template_id,
                "trust": vs.get("trust", "N/A"),
                "verbosity": vs.get("verbosity", "N/A"),
                "pacing": personality.get("PACING", "N/A"),
                "modifiers": personality.get("MODIFIERS", []),
                "episode_density": personality.get("EPISODE_DENSITY", "N/A"),
                "warmth": doctor_microstyle.get("warmth", "N/A"),
                "directness": doctor_microstyle.get("directness", "N/A"),
                "microstyle_pacing": doctor_microstyle.get("pacing", "N/A"),
                "patient_background": patient_background_summary,
                "risk_summary": risk_summary,
            }
            
            # Get doctor persona style description
            doctor_persona_style = doctor_persona.get("system_prompt", "").split('\n')[0] if doctor_persona.get("system_prompt") else "professional primary-care physician"
            
            # Build post-DSM manager input (no DSM list needed)
            from synthetic_datagen.prompts.doctor_manager_prompt import DOCTOR_MANAGER_POST_DSM_SYSTEM_PROMPT
            
            post_dsm_manager_input = f"""Doctor persona:
id: {doctor_persona_id}
style: {doctor_persona_style}
microstyle: warmth={patient_meta['warmth']}, directness={patient_meta['directness']}, pacing={patient_meta['microstyle_pacing']}

Patient background: {patient_meta['patient_background']}
Patient risk summary: {patient_meta['risk_summary']}

Patient profile:
template_id: {patient_meta['template_id']}
trust: {patient_meta['trust']}
verbosity: {patient_meta['verbosity']}
pacing: {patient_meta['pacing']}
modifiers: {patient_meta['modifiers']}
episode_density: {patient_meta['episode_density']}

Conversation so far:
"""
            conversation_context = "\n".join([
                f"{'Doctor' if msg['role'] == 'assistant' else 'Patient'}: {msg['content']}"
                for msg in conversation_history
            ])
            post_dsm_manager_input += conversation_context + "\n\n\nTask: Decide next_action and output JSON only."
            
            # Build post-DSM doctor manager agent
            from synthetic_datagen.generation.manager_logic import parse_post_dsm_manager_output
            
            post_dsm_manager = build_doctor_manager_agent(
                DOCTOR_MANAGER_POST_DSM_SYSTEM_PROMPT,
                get_model_settings(MANAGER_MAX_TOKENS)
            )
            
            # Call post-DSM manager
            post_dsm_session = SQLiteSession(":memory:")
            post_dsm_result = runner.run_sync(post_dsm_manager, post_dsm_manager_input, session=post_dsm_session)
            post_dsm_output = post_dsm_result.final_output
            
            # Track token usage
            if hasattr(post_dsm_result, 'context_wrapper') and hasattr(post_dsm_result.context_wrapper, 'usage'):
                usage = post_dsm_result.context_wrapper.usage
                token_usage["doctor_manager"]["input_tokens"] += usage.input_tokens
                token_usage["doctor_manager"]["output_tokens"] += usage.output_tokens
                token_usage["doctor_manager"]["total_tokens"] += usage.total_tokens
            
            # Log prompt trace
            prompt_traces.append({
                "agent": "doctor_manager_post_dsm",
                "turn_index": doctor_turns_elapsed + 1,
                "system_prompt": DOCTOR_MANAGER_POST_DSM_SYSTEM_PROMPT,
                "input": post_dsm_manager_input,
                "output": post_dsm_output,
            })
            
            # Parse post-DSM manager decision
            post_dsm_decision = parse_post_dsm_manager_output(post_dsm_output)
            next_action = post_dsm_decision["next_action"]
            reason = post_dsm_decision["reason"]
            doctor_instruction = post_dsm_decision["doctor_instruction"]
            
            if log_level != LOG_MINIMAL:
                print(f"    Action: {next_action}")
                print(f"    Reason: {reason}")
                print(f"    Instruction: {doctor_instruction}")
            
            # Store decision
            doctor_manager_decisions.append({
                "turn": doctor_turns_elapsed + 1,
                "manager_type": "post_dsm",
                "next_action": next_action,
                "reason": reason,
                "doctor_instruction": doctor_instruction,
                "dsm_symptom_key": "",
            })
            
            # Handle END action
            if next_action == "END":
                # Generate closing message
                closing_msg = "Thank you for sharing today. Based on what we've discussed, I'll review everything and we can talk about next steps. Is there anything else you'd like to add before we wrap up?"
                print_dialogue_line("doctor", closing_msg, log_level)
                conversation_history.append({"role": "assistant", "content": closing_msg})
                break
            
            # For FOLLOW_UP or RAPPORT, doctor_instruction is already set above
            
        elif remaining_total_turns <= remaining_dsm_items:
            # DSM coverage enforcement: force DSM when no slack (uses total turns including buffer)
            # Use FORCE_DSM prompt to get smooth transitions instead of hardcoded instructions
            if log_level != LOG_MINIMAL:
                print()
                print("=" * 60)
                print(f"  DOCTOR TURN {doctor_turns_elapsed + 1} (FORCE-DSM)")
                print(f"  [Coverage guard: remaining_turns={remaining_total_turns} <= remaining_dsm={remaining_dsm_items}]")
                print("=" * 60)
                print("\n  [MANAGER GUIDANCE]")
            
            # Pick next DSM symptom key
            dsm_symptom_key = dsm_pool[0]  # Take first remaining
            
            # Build input for FORCE_DSM manager
            vs = personality.get("VOICE_STYLE", {})
            patient_meta = {
                "template_id": template_id,
                "trust": vs.get("trust", "N/A"),
                "verbosity": vs.get("verbosity", "N/A"),
                "pacing": personality.get("PACING", "N/A"),
                "modifiers": personality.get("MODIFIERS", []),
                "episode_density": personality.get("EPISODE_DENSITY", "N/A"),
                "warmth": doctor_microstyle.get("warmth", "N/A"),
                "directness": doctor_microstyle.get("directness", "N/A"),
                "microstyle_pacing": doctor_microstyle.get("pacing", "N/A"),
                "patient_background": patient_background_summary,
                "risk_summary": risk_summary,
            }
            
            doctor_persona_style = doctor_persona.get("system_prompt", "").split('\n')[0] if doctor_persona.get("system_prompt") else "professional primary-care physician"
            
            force_dsm_input = f"""Doctor persona:
id: {doctor_persona_id}
style: {doctor_persona_style}
microstyle: warmth={patient_meta['warmth']}, directness={patient_meta['directness']}, pacing={patient_meta['microstyle_pacing']}

Patient background: {patient_meta['patient_background']}

Patient profile:
template_id: {patient_meta['template_id']}
trust: {patient_meta['trust']}
verbosity: {patient_meta['verbosity']}

Required DSM symptom key to ask about: {dsm_symptom_key}

Last patient message:
{patient_reply}

Task: Provide guidance for how this doctor would smoothly transition to asking about {dsm_symptom_key}. Output JSON only."""
            
            # Call FORCE_DSM manager
            force_dsm_manager = build_doctor_manager_agent(
                DOCTOR_MANAGER_FORCE_DSM_SYSTEM_PROMPT,
                get_model_settings(MANAGER_MAX_TOKENS)
            )
            force_dsm_session = SQLiteSession(":memory:")
            force_dsm_result = runner.run_sync(force_dsm_manager, force_dsm_input, session=force_dsm_session)
            force_dsm_output = force_dsm_result.final_output
            
            # Track token usage
            if hasattr(force_dsm_result, 'context_wrapper') and hasattr(force_dsm_result.context_wrapper, 'usage'):
                usage = force_dsm_result.context_wrapper.usage
                token_usage["doctor_manager"]["input_tokens"] += usage.input_tokens
                token_usage["doctor_manager"]["output_tokens"] += usage.output_tokens
                token_usage["doctor_manager"]["total_tokens"] += usage.total_tokens
            
            # Log prompt trace
            prompt_traces.append({
                "agent": "doctor_manager_force_dsm",
                "turn_index": doctor_turns_elapsed + 1,
                "system_prompt": DOCTOR_MANAGER_FORCE_DSM_SYSTEM_PROMPT,
                "input": force_dsm_input,
                "output": force_dsm_output,
            })
            
            # Parse output
            force_dsm_decision = parse_doctor_manager_output(force_dsm_output, dsm_pool)
            next_action = "DSM"  # Always DSM in force mode
            reason = force_dsm_decision.get("reason", f"Forced DSM for {dsm_symptom_key}")
            doctor_instruction = force_dsm_decision.get("doctor_instruction", f"Ask about {dsm_symptom_key} in a natural, conversational way.")
            
            if log_level != LOG_MINIMAL:
                print(f"    Action: {next_action}")
                print(f"    DSM Key: {dsm_symptom_key}")
                print(f"    Instruction: {doctor_instruction}")
            
            # Store forced decision
            doctor_manager_decisions.append({
                "turn": doctor_turns_elapsed + 1,
                "manager_type": "forced_dsm_coverage",
                "next_action": next_action,
                "reason": reason,
                "doctor_instruction": doctor_instruction,
                "dsm_symptom_key": dsm_symptom_key,
            })
            
        else:
            # Ratio-based low-turns mode trigger (uses dsm_turns, not total turns)
            ratio = remaining_dsm_turns / remaining_dsm_items if remaining_dsm_items > 0 else float('inf')
            
            # Determine which system prompt to use based on ratio
            if ratio < target_turns_per_symptom:
                # Low-turns mode: behind schedule
                manager_system_prompt = DOCTOR_MANAGER_LOW_TURNS_SYSTEM_PROMPT
                manager_type = "low_turns"
                if log_level != LOG_MINIMAL:
                    print()
                    print("=" * 60)
                    print(f"  DOCTOR TURN {doctor_turns_elapsed + 1} (LOW-TURNS MODE)")
                    print(f"  [ratio={ratio:.2f} < target={target_turns_per_symptom}]")
                    print("=" * 60)
            else:
                # Normal mode: on schedule or ahead
                manager_system_prompt = DOCTOR_MANAGER_SYSTEM_PROMPT
                manager_type = "normal"
                if log_level != LOG_MINIMAL:
                    print()
                    print("=" * 60)
                    print(f"  DOCTOR TURN {doctor_turns_elapsed + 1} (NORMAL)")
                    print(f"  [ratio={ratio:.2f} >= target={target_turns_per_symptom}]")
                    print("=" * 60)
            
            if log_level != LOG_MINIMAL:
                print("\n  [MANAGER GUIDANCE]")
            
            # Build patient metadata for manager (Step 5: minimal context)
            vs = personality.get("VOICE_STYLE", {})
            patient_meta = {
                "template_id": template_id,
                "trust": vs.get("trust", "N/A"),
                "verbosity": vs.get("verbosity", "N/A"),
                "pacing": personality.get("PACING", "N/A"),
                "modifiers": personality.get("MODIFIERS", []),
                "episode_density": personality.get("EPISODE_DENSITY", "N/A"),
                "warmth": doctor_microstyle.get("warmth", "N/A"),
                "directness": doctor_microstyle.get("directness", "N/A"),
                "microstyle_pacing": doctor_microstyle.get("pacing", "N/A"),
                "patient_background": patient_background_summary,
                "risk_summary": risk_summary,
            }
            
            # Get doctor persona style description
            doctor_persona_style = doctor_persona.get("system_prompt", "").split('\n')[0] if doctor_persona.get("system_prompt") else "professional primary-care physician"
            
            # Build manager input
            manager_input = build_manager_input(
                conversation_history,
                dsm_pool,
                doctor_persona_id,
                doctor_persona_style,
                patient_meta,
                remaining_dsm_turns,
            )
            
            # Add explicit last patient message to manager input
            manager_input = (
                f"{manager_input}\n\n"
                f"Last patient message to respond to:\n{patient_reply}"
            )
            
            # Build doctor manager agent with appropriate prompt (stateless - fresh session per call)
            current_doctor_manager = build_doctor_manager_agent(
                manager_system_prompt,
                get_model_settings(MANAGER_MAX_TOKENS)
            )
            
            # Call doctor manager agent with fresh session (stateless)
            manager_session = SQLiteSession(":memory:")
            doctor_manager_result = runner.run_sync(current_doctor_manager, manager_input, session=manager_session)
            doctor_manager_output = doctor_manager_result.final_output
            
            # Track token usage
            if hasattr(doctor_manager_result, 'context_wrapper') and hasattr(doctor_manager_result.context_wrapper, 'usage'):
                usage = doctor_manager_result.context_wrapper.usage
                token_usage["doctor_manager"]["input_tokens"] += usage.input_tokens
                token_usage["doctor_manager"]["output_tokens"] += usage.output_tokens
                token_usage["doctor_manager"]["total_tokens"] += usage.total_tokens
            
            # Log prompt trace
            prompt_traces.append({
                "agent": "doctor_manager",
                "turn_index": doctor_turns_elapsed + 1,
                "system_prompt": manager_system_prompt,
                "input": manager_input,
                "output": doctor_manager_output,
            })
            
            # Parse doctor manager decision (with fallbacks)
            doctor_manager_decision = parse_doctor_manager_output(doctor_manager_output, dsm_pool)
            next_action = doctor_manager_decision["next_action"]
            reason = doctor_manager_decision["reason"]
            doctor_instruction = doctor_manager_decision["doctor_instruction"]
            dsm_symptom_key = doctor_manager_decision["dsm_symptom_key"]
            
            if log_level != LOG_MINIMAL:
                print(f"    Action: {next_action}" + (f" → {dsm_symptom_key}" if dsm_symptom_key else ""))
                print(f"    Reason: {reason}")
                print(f"    Instruction: {doctor_instruction}")
            
            # Store doctor manager decision
            doctor_manager_decisions.append({
                "turn": doctor_turns_elapsed + 1,
                "manager_type": manager_type,
                "next_action": next_action,
                "reason": reason,
                "doctor_instruction": doctor_instruction,
                "dsm_symptom_key": dsm_symptom_key if next_action == "DSM" else "",
            })
        
        # Increment doctor turns (Step 7: counts extra turns)
        doctor_turns_elapsed += 1
        
        # Build directive tag block based on manager decision
        if next_action == "DSM":
            # Validate and use DSM symptom key
            if dsm_symptom_key in dsm_pool:
                # Valid DSM key
                dsm_pool.remove(dsm_symptom_key)
                asked_question_order.append(dsm_symptom_key)
            else:
                # Fallback if invalid
                if log_level != LOG_MINIMAL:
                    print(f"Warning: Doctor manager selected invalid DSM key '{dsm_symptom_key}'. Using fallback.")
                if dsm_pool:
                    dsm_symptom_key = dsm_pool[0]
                    dsm_pool.remove(dsm_symptom_key)
                    asked_question_order.append(dsm_symptom_key)
                    doctor_instruction = f"Ask about {dsm_symptom_key} in a natural, conversational way."
            
            directive_tag_block = f"<NEXT_QUESTION>\n{doctor_instruction}\n</NEXT_QUESTION>"
            
        elif next_action == "FOLLOW_UP":
            directive_tag_block = f"<FOLLOW_UP>\n{doctor_instruction}\n</FOLLOW_UP>"
            
        else:  # RAPPORT
            directive_tag_block = f"<RAPPORT>\n{doctor_instruction}\n</RAPPORT>"
        
        # Build doctor input with explicit last patient message
        doctor_input = (
            f"Patient just said:\n"
            f'"{patient_reply}"\n\n'
            f"{directive_tag_block}"
        )
        
        # Doctor's turn
        dr = runner.run_sync(doctor_agent, doctor_input, session=doctor_session)
        doctor_reply = dr.final_output
        
        # Track token usage
        if hasattr(dr, 'context_wrapper') and hasattr(dr.context_wrapper, 'usage'):
            usage = dr.context_wrapper.usage
            token_usage["doctor"]["input_tokens"] += usage.input_tokens
            token_usage["doctor"]["output_tokens"] += usage.output_tokens
            token_usage["doctor"]["total_tokens"] += usage.total_tokens
        
        # Log prompt trace
        prompt_traces.append({
            "agent": "doctor",
            "turn_index": doctor_turns_elapsed,
            "system_prompt": doctor_instructions,
            "input": doctor_input,
            "output": doctor_reply,
        })
        
        print_dialogue_line("doctor", doctor_reply, log_level)
        conversation_history.append({"role": "assistant", "content": doctor_reply})
        
        # Store last_doctor_move for patient manager
        last_doctor_move = next_action  # "DSM", "FOLLOW_UP", or "RAPPORT"
        
        # P6: Build patient manager input
        patient_manager_meta = {
            "template_id": template_id,
            "emphasized_symptoms": template.get("emphasized_symptoms", []),
            "modifiers": personality.get("MODIFIERS", []),
            "voice_style": personality.get("VOICE_STYLE", {}),
            "pacing": personality.get("PACING", "MED"),
            "episode_density": personality.get("EPISODE_DENSITY", "MED"),
        }
        
        pm_input = build_patient_manager_input(
            last_doctor_message=doctor_reply,
            last_doctor_move=last_doctor_move,
            conversation_history=conversation_history,
            patient_meta=patient_manager_meta,
            depression_profile=depression_profile,
            disclosure_state=disclosure_state,
        )
        
        # Call patient manager agent with fresh session (stateless)
        if log_level != LOG_MINIMAL:
            print()
            print("-" * 60)
            print(f"  PATIENT TURN {doctor_turns_elapsed}")
            print("-" * 60)
            print("\n  [MANAGER GUIDANCE]")
        pm_session = SQLiteSession(":memory:")
        pm_res = runner.run_sync(patient_manager_agent, pm_input, session=pm_session)
        pm_output_text = pm_res.final_output
        
        # Track token usage
        if hasattr(pm_res, 'context_wrapper') and hasattr(pm_res.context_wrapper, 'usage'):
            usage = pm_res.context_wrapper.usage
            token_usage["patient_manager"]["input_tokens"] += usage.input_tokens
            token_usage["patient_manager"]["output_tokens"] += usage.output_tokens
            token_usage["patient_manager"]["total_tokens"] += usage.total_tokens
        
        # Log prompt trace
        prompt_traces.append({
            "agent": "patient_manager",
            "turn_index": doctor_turns_elapsed,
            "system_prompt": PATIENT_MANAGER_SYSTEM_PROMPT,
            "input": pm_input,
            "output": pm_output_text,
        })
        
        # Parse manager guidance
        guidance = parse_patient_manager_output(
            pm_output_text,
            base_voice_style=personality.get("VOICE_STYLE", {}),
            current_disclosure_state=disclosure_state,
        )
        
        # Get ground truth severity if this was a DSM question
        ground_truth_note = ""
        if last_doctor_move == "DSM" and dsm_symptom_key:
            gt_severity = depression_profile.get(dsm_symptom_key, "N/A")
            ground_truth_note = f" [Ground Truth: {dsm_symptom_key} = {gt_severity}]"
        
        if log_level != LOG_MINIMAL:
            print(f"    Disclosure: {guidance['disclosure_stage']} | Length: {guidance['target_length']}")
            print(f"    Emotional: {guidance.get('emotional_state', 'neutral')}")
            print(f"    Instruction: {guidance['patient_instruction']}")
            if ground_truth_note:
                print(f"    {ground_truth_note}")
        
        # Update disclosure_state
        disclosure_state = guidance["disclosure_stage"]
        
        # Store guidance for metadata
        patient_manager_decisions.append({
            "turn": doctor_turns_elapsed,
            "doctor_move": last_doctor_move,
            "guidance": guidance,
        })
        
        # Build input to patient agent with guidance
        patient_input = f"""PATIENT_GUIDANCE:
directness: {guidance['directness']}
disclosure_stage: {guidance['disclosure_stage']}
target_length: {guidance['target_length']}
emotional_state: {guidance.get('emotional_state', 'neutral')}
tone_tags: {guidance['tone_tags']}
key_points_to_reveal: {guidance['key_points_to_reveal']}
key_points_to_avoid: {guidance['key_points_to_avoid']}
instruction: {guidance['patient_instruction']}

DOCTOR_MESSAGE:
{doctor_reply}
"""
        
        # Patient's turn with guidance
        pr = runner.run_sync(patient_agent, patient_input, session=patient_session)
        patient_reply = pr.final_output
        
        # Track token usage
        if hasattr(pr, 'context_wrapper') and hasattr(pr.context_wrapper, 'usage'):
            usage = pr.context_wrapper.usage
            token_usage["patient"]["input_tokens"] += usage.input_tokens
            token_usage["patient"]["output_tokens"] += usage.output_tokens
            token_usage["patient"]["total_tokens"] += usage.total_tokens
        
        # Log prompt trace
        prompt_traces.append({
            "agent": "patient",
            "turn_index": doctor_turns_elapsed,
            "system_prompt": patient_system_prompt,
            "input": patient_input,
            "output": patient_reply,
        })
        
        print_dialogue_line("patient", patient_reply, log_level)
        conversation_history.append({"role": "user", "content": patient_reply})
    
    # Close if patient spoke last
    if conversation_history[-1]["role"] == "user":
        final_msg = "Thank you for sharing. Is there anything else you'd like to discuss today?"
        print_dialogue_line("doctor", final_msg, log_level)
        dr = runner.run_sync(doctor_agent, final_msg, session=doctor_session)
        
        # Track token usage for final message
        if hasattr(dr, 'context_wrapper') and hasattr(dr.context_wrapper, 'usage'):
            usage = dr.context_wrapper.usage
            token_usage["doctor"]["input_tokens"] += usage.input_tokens
            token_usage["doctor"]["output_tokens"] += usage.output_tokens
            token_usage["doctor"]["total_tokens"] += usage.total_tokens
        
        conversation_history.append({"role": "assistant", "content": final_msg})
    
    # Convert conversation to readable format with explicit speaker names
    readable_conversation = []
    for msg in conversation_history:
        if msg["role"] == "assistant":
            speaker = "doctor"
        elif msg["role"] == "user":
            speaker = "patient"
        else:
            speaker = msg["role"]
        readable_conversation.append({
            "speaker": speaker,
            "text": msg["content"],
        })
    
    # Serialize life_background if present
    life_background_dict = None
    if life_background:
        life_background_dict = {
            "name": life_background.name,
            "age_range": life_background.age_range,
            "pronouns": life_background.pronouns,
            "core_roles": life_background.core_roles,
            "core_relationships": life_background.core_relationships,
            "core_stressor_summary": life_background.core_stressor_summary,
            "life_facets": [
                {
                    "category": facet.category,
                    "salience": facet.salience,
                    "description": facet.description,
                }
                for facet in life_background.life_facets
            ],
        }
    
    # Build session data
    session_data = {
        "run_timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "agent_id": agent_id,
        "template_id": template_id,
        "template_details": template,
        "persona_id": doctor_persona_id,
        "persona_details": doctor_persona,
        "microstyle": doctor_microstyle,
        "personality": personality,
        "life_background": life_background_dict,
        "depression_profile_ground_truth": depression_profile,
        "asked_question_order": asked_question_order,
        "doctor_manager_decisions": doctor_manager_decisions,
        "patient_manager_decisions": patient_manager_decisions,
        "final_disclosure_state": disclosure_state,
        "openai_call_parameters": {
            "model": OPENAI_MODEL,
            "doctor_temperature": doctor_temperature,
            "patient_temperature": patient_temperature,
        },
        "conversation": readable_conversation,
        "raw_conversation": conversation_history,
    }
    
    # Save prompt traces to separate file
    trace_dir = "outputs/prompt_traces"
    os.makedirs(trace_dir, exist_ok=True)
    trace_fname = os.path.join(trace_dir, f"prompttrace_{agent_id}.json")
    
    with open(trace_fname, "w", encoding="utf-8") as f:
        json.dump({
            "agent_id": agent_id,
            "prompt_traces": prompt_traces,
        }, f, indent=2)
    
    session_data["prompt_trace_file"] = trace_fname
    
    # Save raw log file with full details
    log_dir = "outputs/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_fname = os.path.join(log_dir, f"session_{agent_id}_raw.json")
    
    raw_log_data = {
        "agent_id": agent_id,
        "session_data": session_data,
        "prompt_traces": prompt_traces,
    }
    
    with open(log_fname, "w", encoding="utf-8") as f:
        json.dump(raw_log_data, f, indent=2)
    
    session_data["raw_log_file"] = log_fname
    session_data["token_usage"] = token_usage
    
    # Print token usage summary
    if log_level != LOG_MINIMAL:
        print(f"\n{'=' * 60}")
        print("=== TOKEN USAGE SUMMARY ===")
        print(f"{'=' * 60}")
        total_tokens = sum(agent["total_tokens"] for agent in token_usage.values())
        total_input = sum(agent["input_tokens"] for agent in token_usage.values())
        total_output = sum(agent["output_tokens"] for agent in token_usage.values())
        
        for agent_name, usage in token_usage.items():
            print(f"{agent_name}:")
            print(f"  Input:  {usage['input_tokens']:,} tokens")
            print(f"  Output: {usage['output_tokens']:,} tokens")
            print(f"  Total:  {usage['total_tokens']:,} tokens")
        
        print(f"\nGrand Total:")
        print(f"  Input:  {total_input:,} tokens")
        print(f"  Output: {total_output:,} tokens")
        print(f"  Total:  {total_tokens:,} tokens")
        print(f"{'=' * 60}")
        
        print(f"\nSession complete: {agent_id}")
        print(f"Transcript: outputs/transcripts/transcript_{agent_id}.json")
        print(f"Raw log: {log_fname}")
    
    return session_data
