"""
Pure functions for patient profile generation.
"""
import random
from typing import Dict, Any, Optional

from synthetic_datagen.data.templates import BIG5_DEP_TEMPLATES, DSM5_DEPRESSION_SYMPTOMS
from synthetic_datagen.data.pools import (
    PACING_LEVELS,
    GENERIC_CONTEXT_DOMAINS,
    MAX_CONTEXT_DOMAINS,
    INTENSITY_LEVELS,
    EPISODE_DENSITY_LEVELS,
    VOICE_STYLE_OPTIONS,
    PATIENT_HUMOR_LEVELS,
    LIVING_SITUATION_POOL,
    WORK_ROLE_POOL,
    ROUTINE_STABILITY_POOL,
    SUPPORT_LEVEL_POOL,
    AGE_RANGES,
    AGE_DEFAULT_WEIGHTS,
    AGE_WEIGHTS_BY_ROLE,
)
from synthetic_datagen.generation.background_writer import call_background_writer


def generate_depression_profile(
    rng: random.Random,
    template: Dict[str, Any],
    episode_density: str,
    emph_intensity: Dict[str, str],
    extra_high: list,
) -> Dict[str, str]:
    """
    Generate depression symptom frequency profile based on template and parameters.
    
    Parameters
    ----------
    rng : random.Random
        Random number generator instance
    template : dict
        Big Five template dict containing emphasized_symptoms
    episode_density : str
        One of ULTRA_LOW, LOW, MED, HIGH
    emph_intensity : dict
        Mapping of emphasized symptom names to LOW/MED/HIGH intensity
    extra_high : list
        List of non-emphasized symptoms to elevate
        
    Returns
    -------
    dict
        Mapping of symptom name to frequency code (NONE/RARE/SOME/OFTEN)
    """
    depression_profile = {}
    emphasized = set(template["emphasized_symptoms"])
    
    # ULTRA_LOW mode: override to allow 0-2 symptoms total
    if episode_density == "ULTRA_LOW":
        # Choose target symptom count: 0, 1, or 2
        target_symptom_count = rng.choice([0, 1, 2])
        
        # Build candidate pool: prefer emphasized symptoms first
        candidate_pool = list(emphasized)
        
        # If target exceeds emphasized symptoms, add non-emphasized
        non_emphasized = [s for s in DSM5_DEPRESSION_SYMPTOMS if s not in emphasized]
        if target_symptom_count > len(candidate_pool):
            candidate_pool.extend(non_emphasized)
        
        # Select symptoms to be non-NONE
        if target_symptom_count > 0:
            selected_symptoms = rng.sample(candidate_pool, k=min(target_symptom_count, len(candidate_pool)))
        else:
            selected_symptoms = []
        
        # Initialize all symptoms to NONE
        for symptom in DSM5_DEPRESSION_SYMPTOMS:
            depression_profile[symptom] = "NONE"
        
        # Assign non-NONE frequencies to selected symptoms
        for symptom in selected_symptoms:
            if symptom in emphasized:
                # Use intensity for emphasized symptoms
                level = emph_intensity[symptom]
                if level == "LOW":
                    depression_profile[symptom] = rng.choices(
                        ["RARE", "SOME", "OFTEN"], weights=[5, 4, 1], k=1
                    )[0]
                elif level == "HIGH":
                    depression_profile[symptom] = rng.choices(
                        ["SOME", "OFTEN"], weights=[2, 8], k=1
                    )[0]
                else:  # MED
                    depression_profile[symptom] = rng.choices(
                        ["SOME", "OFTEN", "RARE"], weights=[5, 4, 1], k=1
                    )[0]
            else:
                # For non-emphasized: lighter elevation
                depression_profile[symptom] = rng.choices(
                    ["RARE", "SOME", "OFTEN"], weights=[5, 3, 2], k=1
                )[0]
    else:
        # Standard density modes: original logic
        # Define density-based weights for non-emphasized symptoms
        if episode_density == "LOW":
            non_emph_weights = (7, 2, 1)  # NONE, RARE, SOME
        elif episode_density == "HIGH":
            non_emph_weights = (3, 3, 4)
        else:  # MED
            non_emph_weights = (5, 3, 2)
        
        for symptom in DSM5_DEPRESSION_SYMPTOMS:
            if symptom in emphasized:
                # Apply intensity-based sampling for emphasized symptoms
                level = emph_intensity[symptom]
                if level == "LOW":
                    depression_profile[symptom] = rng.choices(
                        ["RARE", "SOME", "OFTEN"], weights=[5, 4, 1], k=1
                    )[0]
                elif level == "HIGH":
                    depression_profile[symptom] = rng.choices(
                        ["SOME", "OFTEN"], weights=[2, 8], k=1
                    )[0]
                else:  # MED
                    depression_profile[symptom] = rng.choices(
                        ["SOME", "OFTEN", "RARE"], weights=[5, 4, 1], k=1
                    )[0]
            elif symptom in extra_high:
                # Treat as secondary emphasized (lighter elevation)
                depression_profile[symptom] = rng.choices(
                    ["RARE", "SOME", "OFTEN"], weights=[2, 5, 3], k=1
                )[0]
            else:
                # Non-emphasized: use density-based weights
                depression_profile[symptom] = rng.choices(
                    ["NONE", "RARE", "SOME"], weights=non_emph_weights, k=1
                )[0]
    
    return depression_profile


def sample_patient_profile(rng: random.Random, forced: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Sample a complete patient profile with personality and depression characteristics.
    
    This is a pure function that uses the provided RNG for all random choices,
    allowing for reproducible testing and controlled generation.
    
    Parameters
    ----------
    rng : random.Random
        Random number generator instance for reproducibility
    forced : dict, optional
        Dictionary with forced overrides for specific parameters:
        - template_id: force specific Big Five template
        - episode_density: force specific density level
        - Any other personality/profile attributes
        
    Returns
    -------
    dict
        Complete profile with keys:
        - template_id: str
        - template: dict
        - personality: dict (with modifiers, pacing, contexts, voice, background, etc.)
        - depression_profile: dict (symptom -> frequency mapping)
    """
    forced = forced or {}
    
    # Select template
    if "template_id" in forced:
        template_id = forced["template_id"]
    else:
        template_id = rng.choice(list(BIG5_DEP_TEMPLATES.keys()))
    
    template = BIG5_DEP_TEMPLATES[template_id]
    
    # Build personality dict
    personality = {
        "BIG5_TEMPLATE": template_id,
        "AFFECTIVE_STYLE": template["affective"],
        "COGNITIVE_STYLE": template["cognitive"],
        "SOMATIC_STYLE": template["somatic"],
        "SPECIFIER_HINT": template["specifier"],
    }
    
    # Sample template modifiers (0-2)
    if "modifiers" in forced:
        modifiers = forced["modifiers"]
    else:
        modifiers = rng.sample(template["modifiers"], k=rng.randint(0, 2))
    personality["MODIFIERS"] = modifiers
    
    # Sample conversation pacing
    if "pacing" in forced:
        pacing = forced["pacing"]
    else:
        pacing = rng.choice(PACING_LEVELS)
    personality["PACING"] = pacing
    
    # Sample context domains (0-2)
    if "context_domains" in forced:
        context_domains = forced["context_domains"]
    else:
        num_domains = rng.randint(0, MAX_CONTEXT_DOMAINS)
        context_domains = rng.sample(GENERIC_CONTEXT_DOMAINS, k=num_domains)
    personality["CONTEXT_DOMAINS"] = context_domains
    
    # Sample episode density with ULTRA_LOW (~10%), LOW (~25%), MED (~45%), HIGH (~20%)
    if "episode_density" in forced:
        episode_density = forced["episode_density"]
    else:
        episode_density = rng.choices(
            EPISODE_DENSITY_LEVELS, weights=[1, 2.5, 4.5, 2], k=1
        )[0]
    personality["EPISODE_DENSITY"] = episode_density
    
    # Sample voice style dimensions independently
    if "voice_style" in forced:
        voice_style = forced["voice_style"]
    else:
        voice_style = {
            dim: rng.choice(options)
            for dim, options in VOICE_STYLE_OPTIONS.items()
        }
        # Add humor (weighted: 60% none, 30% occasional, 10% frequent)
        voice_style["humor"] = rng.choices(
            PATIENT_HUMOR_LEVELS,
            weights=[6, 3, 1],
            k=1
        )[0]
    
    # Apply partial voice style overrides (specific dimensions only)
    if "voice_style_partial" in forced:
        for dim, value in forced["voice_style_partial"].items():
            voice_style[dim] = value
    
    personality["VOICE_STYLE"] = voice_style
    
    # Sample personal background (2-3 facts)
    if "personal_background" in forced:
        personal_background = forced["personal_background"]
        work_role = personal_background.get("work_role", "employed")
    else:
        work_role = rng.choice(WORK_ROLE_POOL)
        personal_background = {
            "living_situation": rng.choice(LIVING_SITUATION_POOL),
            "work_role": work_role,
            "routine_stability": rng.choice(ROUTINE_STABILITY_POOL),
            "support_level": rng.choice(SUPPORT_LEVEL_POOL),
        }
        # Drop 1-2 fields randomly so only 2-3 remain
        keys = list(personal_background.keys())
        for k in rng.sample(keys, k=rng.randint(1, 2)):
            personal_background.pop(k)
    personality["PERSONAL_BACKGROUND"] = personal_background
    
    # Sample age range based on work role
    if "age_range" in forced:
        age_range = forced["age_range"]
    else:
        # Get role-appropriate age weights
        age_weights = AGE_WEIGHTS_BY_ROLE.get(work_role, AGE_DEFAULT_WEIGHTS)
        age_range = rng.choices(AGE_RANGES, weights=age_weights, k=1)[0]
    personality["AGE_RANGE"] = age_range
    
    # Sample intensity per emphasized symptom
    emphasized = set(template["emphasized_symptoms"])
    if "emph_intensity" in forced:
        emph_intensity = forced["emph_intensity"]
    else:
        emph_intensity = {}
        for sym in emphasized:
            emph_intensity[sym] = rng.choices(
                INTENSITY_LEVELS, weights=[3, 5, 2], k=1
            )[0]
    personality["EMPH_INTENSITY"] = emph_intensity
    
    # Select 0-1 non-emphasized symptoms for extra elevation
    non_emphasized = [s for s in DSM5_DEPRESSION_SYMPTOMS if s not in emphasized]
    if "extra_elevated" in forced:
        extra_high = forced["extra_elevated"]
    else:
        extra_high = rng.sample(non_emphasized, k=rng.randint(0, 1))
    personality["EXTRA_ELEVATED_SYMPTOMS"] = extra_high
    
    # Generate depression profile
    depression_profile = generate_depression_profile(
        rng, template, episode_density, emph_intensity, extra_high
    )
    
    # Generate rich life background using background writer
    life_background = None
    background_writer_prompt_trace = None
    if "skip_life_background" not in forced or not forced["skip_life_background"]:
        # Call background writer with complete profile (including age_range)
        bg_result = call_background_writer(
            rng=rng,
            personality=personality,
            depression_profile=depression_profile,
            basic_background=personal_background,
            context_domains=context_domains,
            age_range=age_range,
        )
        life_background = bg_result.get("background")
        background_writer_prompt_trace = bg_result.get("prompt_trace")
    
    return {
        "template_id": template_id,
        "template": template,
        "personality": personality,
        "depression_profile": depression_profile,
        "life_background": life_background,
        "background_writer_prompt_trace": background_writer_prompt_trace,
    }
