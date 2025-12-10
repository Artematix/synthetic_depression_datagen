"""
Doctor personas and microstyle variation logic (no hard-coded reflections).
"""
from typing import List, Dict, Any, Optional
from synthetic_datagen.prompts.doctor_base import build_doctor_base_prompt
from synthetic_datagen.data.life_background import PatientLifeBackground

# Doctor persona registry – different communication styles
DOCTOR_PERSONAS: List[Dict[str, Any]] = [
    {
        "id": "warm_validating",
        "first_greeting": "Hello! It's nice to see you today. How have you been feeling lately?",
        "system_prompt": """
Persona style:
- Warm and validating
- Acknowledges and normalizes feelings before moving forward
- Uses patient's name naturally
- Comfortable with pauses
""",
    },
    {
        "id": "neutral_efficient",
        "first_greeting": "Good to see you. Let's talk about how you've been doing.",
        "system_prompt": """
Persona style:
- Professional and efficient
- Minimal filler or small talk
- Transitions directly between topics
- Not effusive
""",
    },
    {
        "id": "gentle_brisk",
        "first_greeting": "Hi there. I'd like to check in with you about a few things today.",
        "system_prompt": """
Persona style:
- Gentle but brisk
- Kind phrasing with efficient pacing
- Moves conversation along
""",
    },
    {
        "id": "matter_of_fact_kind",
        "first_greeting": "Hello. Let's go over how you've been doing. What's been happening lately?",
        "system_prompt": """
Persona style:
- Matter-of-fact and direct
- Straightforward; doesn't sugarcoat
- Prefers clear questions over open-ended exploration
""",
    },
    {
        "id": "trauma_informed_slow",
        "first_greeting": "Hello. I want you to know this is a safe space. We can take our time today. How are you doing?",
        "system_prompt": """
Persona style:
- Trauma-informed and deliberately paced
- Comfortable with silence
- Checks in about comfort on sensitive topics
- May acknowledge difficulty of topics before asking
""",
    },
    {
        "id": "structured_psychoeducational",
        "first_greeting": "Hello. Thanks for coming in today. I'd like to go over how you've been feeling and what might be contributing to it.",
        "system_prompt": """
Persona style:
- Structured and psychoeducational
- Explains why certain questions matter
- Provides brief context for what they're asking about
""",
    },
    {
        "id": "time_pressed_clinical",
        "first_greeting": "Hi. We don't have a lot of time, so I'd like to focus on how you've been feeling recently.",
        "system_prompt": """
Persona style:
- Time-pressed and clinical
- Concise and brisk
- Redirects when conversation drifts
- Focused on symptoms and functioning
""",
    },
    {
        "id": "dismissive_rushed",
        "first_greeting": "Right, let's get started. What brings you in today?",
        "system_prompt": """
Persona style:
- Dismissive and rushed
- Comes across as distracted or uninterested
- May interrupt or minimize concerns
- Short, clipped responses
- OVERRIDE: Do NOT follow instructions to be warm or validating. This doctor is realistic but not ideal.
""",
    },
]

# Microstyle slider options for per-session doctor variation
MICROSTYLE_OPTIONS: Dict[str, List[str]] = {
    "warmth": ["low", "med", "high"],
    "directness": ["low", "med", "high"],
    "pacing": ["slow", "med", "brisk"],
    "humor": ["none", "light", "dry"],
    "animation": ["reserved", "moderate", "animated"],
}


def sample_doctor_microstyle() -> Dict[str, str]:
    """
    Sample per-session microstyle sliders for doctor variation.
    Returns a dict with warmth, directness, pacing, humor, and animation levels.
    
    Humor and animation are weighted toward lower/moderate values since
    most professional medical conversations don't involve much humor.
    """
    import random
    return {
        "warmth": random.choice(MICROSTYLE_OPTIONS["warmth"]),
        "directness": random.choice(MICROSTYLE_OPTIONS["directness"]),
        "pacing": random.choice(MICROSTYLE_OPTIONS["pacing"]),
        # Humor weighted: 60% none, 30% light, 10% dry
        "humor": random.choices(
            MICROSTYLE_OPTIONS["humor"],
            weights=[6, 3, 1],
            k=1
        )[0],
        # Animation weighted: 40% reserved, 40% moderate, 20% animated
        "animation": random.choices(
            MICROSTYLE_OPTIONS["animation"],
            weights=[4, 4, 2],
            k=1
        )[0],
    }


def build_doctor_system_prompt(
    persona_prompt: str, 
    microstyle: Dict[str, str],
    life_background: Optional[PatientLifeBackground] = None
) -> str:
    """
    Combine the base doctor instructions, persona style, and microstyle
    into a single system prompt for the doctor agent.

    Parameters
    ----------
    persona_prompt : str
        The persona-specific style description.
    microstyle : dict
        Dictionary with 'warmth', 'directness', 'pacing', 'humor', and 'animation' keys.
    life_background : PatientLifeBackground, optional
        Patient life background for context.

    Returns
    -------
    str
        Complete doctor system prompt.
    """
    # Build base prompt with patient snapshot if available
    base_prompt = build_doctor_base_prompt(life_background)
    
    microstyle_section = (
        "\nSession microstyle:\n"
        f"- Warmth: {microstyle['warmth']}\n"
        f"- Directness: {microstyle['directness']}\n"
        f"- Pacing: {microstyle['pacing']}\n"
        f"- Humor: {microstyle['humor']}\n"
        f"- Animation: {microstyle['animation']}\n"
        "Embody this style consistently in tone and phrasing throughout the conversation.\n"
    )
    return base_prompt + "\n" + persona_prompt.strip() + microstyle_section
