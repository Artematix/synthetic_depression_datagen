"""
Background writer logic for generating patient life contexts.
"""
import json
import random
from typing import Dict, List, Any, Optional

from agents import Agent, ModelSettings, Runner, SQLiteSession
from synthetic_datagen.config import OPENAI_MODEL, BACKGROUND_WRITER_TEMPERATURE, BACKGROUND_WRITER_MAX_TOKENS


def get_background_writer_model_settings() -> ModelSettings:
    """
    Get ModelSettings for background writer agent.
    
    Note: gpt-5-mini does not support temperature parameter.
    For gpt-5-mini, we omit model_settings entirely to use defaults.
    """
    if OPENAI_MODEL == "gpt-5-mini":
        # Return None to use default model settings - gpt-5-mini may have issues with explicit ModelSettings
        return None
    else:
        return ModelSettings(
            temperature=BACKGROUND_WRITER_TEMPERATURE,
            max_tokens=BACKGROUND_WRITER_MAX_TOKENS,
        )
from synthetic_datagen.data.pools import (
    LIFE_FACET_CATEGORIES,
    TRAUMA_ADVERSITY_FACETS,
    GENERIC_CONTEXT_DOMAINS,
)
from synthetic_datagen.data.life_background import PatientLifeBackground, LifeFacet
from synthetic_datagen.prompts.background_writer_prompt import BACKGROUND_WRITER_SYSTEM_PROMPT


def compute_symptom_severity(depression_profile: Dict[str, str]) -> str:
    """
    Compute overall symptom severity from depression profile.
    
    Parameters
    ----------
    depression_profile : dict
        Mapping of symptom name to frequency code (NONE/RARE/SOME/OFTEN)
        
    Returns
    -------
    str
        "minimal", "mild", "moderate", or "severe"
    """
    # Count symptoms by frequency
    counts = {"NONE": 0, "RARE": 0, "SOME": 0, "OFTEN": 0}
    for freq in depression_profile.values():
        counts[freq] = counts.get(freq, 0) + 1
    
    # Count significant symptoms (SOME or OFTEN)
    significant_count = counts["SOME"] + counts["OFTEN"]
    often_count = counts["OFTEN"]
    
    # Classify severity
    if significant_count == 0:
        return "minimal"
    elif significant_count <= 2 or (significant_count <= 4 and often_count == 0):
        return "mild"
    elif significant_count <= 5 or often_count <= 3:
        return "moderate"
    else:
        return "severe"


def select_required_facets(
    rng: random.Random,
    context_domains: List[str],
    depression_profile: Dict[str, str],
    severity: str,
) -> List[str]:
    """
    Select 5-8 required facet categories based on context and severity.
    
    Parameters
    ----------
    rng : random.Random
        Random number generator
    context_domains : list
        List of context domain strings
    depression_profile : dict
        Symptom frequency mapping
    severity : str
        Overall severity level
        
    Returns
    -------
    list
        List of 5-8 facet category IDs
    """
    # Build weighted category pool
    category_weights = {cat: 1.0 for cat in LIFE_FACET_CATEGORIES}
    
    # Boost categories related to context domains
    for domain in context_domains:
        if "work" in domain or "role" in domain:
            category_weights["work_or_study_pressure"] = 3.0
            category_weights["sense_of_achievement"] = 2.0
            category_weights["role_conflicts"] = 2.0
            category_weights["responsibility_load"] = 2.0
        
        if "relationship" in domain:
            category_weights["family_relationship_pattern"] = 2.5
            category_weights["closest_friend_or_confidant"] = 2.0
            category_weights["key_partner_or_love_interest"] = 2.0
            category_weights["conflictual_relationship"] = 2.0
        
        if "health" in domain:
            category_weights["physical_health_constraints"] = 3.0
            category_weights["sleep_pattern_tendency"] = 2.0
            category_weights["body_image_concerns_or_comfort"] = 1.5
        
        if "self-worth" in domain or "identity" in domain:
            category_weights["self_view"] = 2.5
            category_weights["beliefs_about_self_worth"] = 2.5
            category_weights["identity_stage"] = 2.0
        
        if "transition" in domain:
            category_weights["significant_move_or_transition"] = 2.5
            category_weights["stalled_goal"] = 2.0
            category_weights["loss_or_change"] = 2.0
        
        if "grief" in domain or "bereavement" in domain:
            category_weights["loss_or_change"] = 3.0
            category_weights["unresolved_issue"] = 2.0
    
    # Always include core stressor-related categories
    category_weights["current_primary_stressor"] = 4.0
    category_weights["coping_style"] = 2.5
    
    # Boost trauma/adversity categories for moderate/severe cases
    if severity in ["moderate", "severe"] or any(d in ["grief/bereavement", "major life transition"] for d in context_domains):
        for trauma_cat in TRAUMA_ADVERSITY_FACETS:
            if trauma_cat in category_weights:
                category_weights[trauma_cat] = category_weights[trauma_cat] * 1.5
    else:
        # Reduce trauma weights for mild cases
        for trauma_cat in TRAUMA_ADVERSITY_FACETS:
            if trauma_cat in category_weights:
                category_weights[trauma_cat] = category_weights[trauma_cat] * 0.5
    
    # Sample 5-8 categories
    num_facets = rng.randint(5, 8)
    
    # Ensure at most 2 trauma/adversity facets in required set
    trauma_in_required = 0
    selected = []
    
    # Convert to list for weighted sampling
    categories = list(category_weights.keys())
    weights = [category_weights[cat] for cat in categories]
    
    attempts = 0
    while len(selected) < num_facets and attempts < 100:
        attempts += 1
        candidate = rng.choices(categories, weights=weights, k=1)[0]
        
        if candidate in selected:
            continue
        
        # Check trauma limit
        if candidate in TRAUMA_ADVERSITY_FACETS:
            if trauma_in_required >= 2:
                continue
            trauma_in_required += 1
        
        selected.append(candidate)
    
    return selected


def build_background_writer_input(
    personality: Dict[str, Any],
    depression_profile: Dict[str, str],
    basic_background: Dict[str, str],
    context_domains: List[str],
    required_facets: List[str],
    age_range: str,
) -> str:
    """
    Build input for background writer agent.
    
    Parameters
    ----------
    personality : dict
        Personality profile with template_id, modifiers, voice_style, etc.
    depression_profile : dict
        Symptom frequency mapping
    basic_background : dict
        Basic tags (living_situation, work_role, routine_stability, support_level)
    context_domains : list
        Selected context domains
    required_facets : list
        List of required facet category IDs
    age_range : str
        Age range string (e.g., "25-29", "40-49")
        
    Returns
    -------
    str
        Formatted input string
    """
    # Format personality summary
    template_id = personality.get("BIG5_TEMPLATE", "N/A")
    modifiers = personality.get("MODIFIERS", [])
    voice_style = personality.get("VOICE_STYLE", {})
    pacing = personality.get("PACING", "N/A")
    episode_density = personality.get("EPISODE_DENSITY", "N/A")
    
    personality_summary = f"""template_id: {template_id}
modifiers: {modifiers}
voice_style: verbosity={voice_style.get('verbosity', 'N/A')}, expressiveness={voice_style.get('expressiveness', 'N/A')}, trust={voice_style.get('trust', 'N/A')}, intellect={voice_style.get('intellect', 'N/A')}
pacing: {pacing}
episode_density: {episode_density}"""
    
    # Format depression profile
    dep_lines = [f"{symptom}: {freq}" for symptom, freq in depression_profile.items()]
    dep_profile_str = "\n".join(dep_lines)
    
    # Format basic background
    bg_lines = [f"{k}: {v}" for k, v in basic_background.items()]
    bg_str = "\n".join(bg_lines)
    
    # Format context domains
    domains_str = ", ".join(context_domains) if context_domains else "none specified"
    
    # Format required facets
    required_str = "\n".join([f"- {cat}" for cat in required_facets])
    
    # Format all facets
    all_str = "\n".join([f"- {cat}" for cat in LIFE_FACET_CATEGORIES])
    
    return f"""Age range: {age_range}

Personality summary:
{personality_summary}

Depression symptom profile:
{dep_profile_str}

Basic background tags:
{bg_str}

Context domains:
{domains_str}

Required facets (you must fill all of these):
{required_str}

All available facets (you may add 2-5 extra from this list):
{all_str}

Task: Generate the patient life background and output JSON only."""


def parse_background_writer_output(output: str) -> Optional[PatientLifeBackground]:
    """
    Parse background writer JSON output into PatientLifeBackground object.
    
    Parameters
    ----------
    output : str
        Raw JSON output from background writer
        
    Returns
    -------
    PatientLifeBackground or None
        Parsed background object, or None if parsing fails
    """
    try:
        # Strip markdown code fences if present
        cleaned = output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        # Parse JSON
        data = json.loads(cleaned)
        
        # Parse life facets
        life_facets = []
        for facet_data in data.get("life_facets", []):
            life_facets.append(LifeFacet(
                category=facet_data.get("category", ""),
                salience=facet_data.get("salience", "med"),
                description=facet_data.get("description", ""),
            ))
        
        # Build PatientLifeBackground
        return PatientLifeBackground(
            name=data.get("name", "Patient"),
            age_range=data.get("age_range", "adult"),
            pronouns=data.get("pronouns", "they/them"),
            core_roles=data.get("core_roles", []),
            core_relationships=data.get("core_relationships", []),
            core_stressor_summary=data.get("core_stressor_summary", ""),
            life_facets=life_facets,
        )
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Failed to parse background writer output: {e}")
        return None


def call_background_writer(
    rng: random.Random,
    personality: Dict[str, Any],
    depression_profile: Dict[str, str],
    basic_background: Dict[str, str],
    context_domains: List[str],
    age_range: str,
) -> Dict[str, Any]:
    """
    Call the background writer agent to generate patient life background.
    
    Parameters
    ----------
    rng : random.Random
        Random number generator
    personality : dict
        Personality profile
    depression_profile : dict
        Symptom frequency mapping
    basic_background : dict
        Basic background tags
    context_domains : list
        Context domains
    age_range : str
        Age range string (e.g., "25-29", "40-49")
        
    Returns
    -------
    dict
        Dictionary containing:
        - "background": PatientLifeBackground or None
        - "prompt_trace": dict with system_prompt, input, output for logging
    """
    # Compute severity
    severity = compute_symptom_severity(depression_profile)
    
    # Select required facets
    required_facets = select_required_facets(
        rng, context_domains, depression_profile, severity
    )
    
    # Build input (including age_range)
    writer_input = build_background_writer_input(
        personality, depression_profile, basic_background, context_domains, required_facets, age_range
    )
    
    # Create agent
    model_settings = get_background_writer_model_settings()
    if model_settings is not None:
        background_writer = Agent(
            name="BackgroundWriter",
            model=OPENAI_MODEL,
            instructions=BACKGROUND_WRITER_SYSTEM_PROMPT,
            model_settings=model_settings,
        )
    else:
        # For gpt-5-mini, don't pass model_settings to avoid issues
        background_writer = Agent(
            name="BackgroundWriter",
            model=OPENAI_MODEL,
            instructions=BACKGROUND_WRITER_SYSTEM_PROMPT,
        )
    
    # Call agent
    session = SQLiteSession(":memory:")
    runner = Runner()
    
    try:
        result = runner.run_sync(background_writer, writer_input, session=session)
        output = result.final_output
        
        # Parse output
        background = parse_background_writer_output(output)
        
        # Build prompt trace for logging
        prompt_trace = {
            "agent": "background_writer",
            "turn_index": 0,
            "system_prompt": BACKGROUND_WRITER_SYSTEM_PROMPT,
            "input": writer_input,
            "output": output,
        }
        
        return {
            "background": background,
            "prompt_trace": prompt_trace,
        }
        
    except Exception as e:
        print(f"Warning: Background writer failed: {e}")
        return {
            "background": None,
            "prompt_trace": {
                "agent": "background_writer",
                "turn_index": 0,
                "system_prompt": BACKGROUND_WRITER_SYSTEM_PROMPT,
                "input": writer_input,
                "output": f"ERROR: {str(e)}",
            },
        }
