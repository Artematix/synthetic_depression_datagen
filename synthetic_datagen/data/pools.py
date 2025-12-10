"""
Data pools for patient profile generation (pacing, contexts, intensity, voice styles, etc.).
"""
from typing import List, Dict

# Conversation pacing levels (patient elaboration style)
PACING_LEVELS: List[str] = ["LOW", "MED", "HIGH"]

# Generic context domains for life situation framing
GENERIC_CONTEXT_DOMAINS: List[str] = [
    "work/role strain",
    "relationships strain",
    "health concern",
    "self-worth/identity strain",
    "general stress/no clear trigger",
    "major life transition",
    "grief/bereavement",
]
MAX_CONTEXT_DOMAINS: int = 2

# Symptom intensity levels (for emphasized symptoms)
INTENSITY_LEVELS: List[str] = ["LOW", "MED", "HIGH"]

# Episode density levels (overall symptom sparsity)
EPISODE_DENSITY_LEVELS: List[str] = ["ULTRA_LOW", "LOW", "MED", "HIGH"]

# Patient humor style (sampled separately from voice style)
# Many patients use humor as a coping mechanism, especially when nervous or deflecting
PATIENT_HUMOR_LEVELS: List[str] = ["none", "occasional", "frequent"]

# Voice style dimension options (sampled independently for more variety)
VOICE_STYLE_OPTIONS: Dict[str, List[str]] = {
    "verbosity": ["terse", "moderate", "detailed"],
    "expressiveness": ["flat", "balanced", "intense"],
    "trust": ["guarded", "neutral", "open"],
    "intellect": ["low-functioning", "moderate-functioning", "high-functioning"],
}

# Legacy: Pre-defined combinations (kept for backwards compatibility if needed)
VOICE_STYLE_PROFILES: List[Dict[str, str]] = [
    {"verbosity": "terse", "expressiveness": "flat", "trust": "guarded", "intellect": "low-functioning"},
    {"verbosity": "terse", "expressiveness": "flat", "trust": "neutral", "intellect": "low-functioning"},
    {"verbosity": "terse", "expressiveness": "balanced", "trust": "guarded", "intellect": "moderate-functioning"},
    {"verbosity": "terse", "expressiveness": "intense", "trust": "open", "intellect": "high-functioning"},
    {"verbosity": "moderate", "expressiveness": "flat", "trust": "guarded", "intellect": "moderate-functioning"},
    {"verbosity": "moderate", "expressiveness": "balanced", "trust": "neutral", "intellect": "moderate-functioning"},
    {"verbosity": "moderate", "expressiveness": "balanced", "trust": "open", "intellect": "high-functioning"},
    {"verbosity": "moderate", "expressiveness": "intense", "trust": "open", "intellect": "high-functioning"},
    {"verbosity": "detailed", "expressiveness": "flat", "trust": "neutral", "intellect": "high-functioning"},
    {"verbosity": "detailed", "expressiveness": "balanced", "trust": "neutral", "intellect": "high-functioning"},
    {"verbosity": "detailed", "expressiveness": "balanced", "trust": "open", "intellect": "high-functioning"},
    {"verbosity": "detailed", "expressiveness": "intense", "trust": "open", "intellect": "high-functioning"},
    {"verbosity": "terse", "expressiveness": "flat", "trust": "neutral", "intellect": "moderate-functioning"},
    {"verbosity": "moderate", "expressiveness": "flat", "trust": "neutral", "intellect": "low-functioning"},
    {"verbosity": "moderate", "expressiveness": "intense", "trust": "guarded", "intellect": "moderate-functioning"},
    {"verbosity": "detailed", "expressiveness": "intense", "trust": "neutral", "intellect": "moderate-functioning"},
]

# Personal background seed pools (generic, non-eventful)
LIVING_SITUATION_POOL: List[str] = ["alone", "with partner", "with family", "shared housing"]
WORK_ROLE_POOL: List[str] = ["employed", "student", "caregiving role", "between roles"]
ROUTINE_STABILITY_POOL: List[str] = ["stable routine", "variable routine"]
SUPPORT_LEVEL_POOL: List[str] = ["low support", "moderate support", "high support"]

# Age ranges (in years) and default weights
AGE_RANGES: List[str] = [
    "16-19",   # late teens
    "20-24",   # early 20s
    "25-29",   # late 20s
    "30-34",   # early 30s
    "35-39",   # late 30s
    "40-49",   # 40s
    "50-59",   # 50s
    "60-69",   # 60s
    "70-80",   # 70s+
]

# Default weights (can be overridden by role)
AGE_DEFAULT_WEIGHTS: List[float] = [1, 2, 2.5, 2, 1.5, 1.5, 1, 0.8, 0.5]

# Role-based age weight adjustments
# Each role has custom weights for the 9 age ranges
AGE_WEIGHTS_BY_ROLE: Dict[str, List[float]] = {
    # Students tend to be younger (16-29 heavily weighted)
    "student": [4, 5, 3, 1, 0.5, 0.3, 0.2, 0.1, 0.05],
    
    # Employed spread across working ages (20-65)
    "employed": [0.5, 2, 3, 3, 3, 2.5, 2, 1, 0.3],
    
    # Caregivers tend to be older (35+), but not exclusively
    "caregiving role": [0.1, 0.3, 0.8, 1.5, 2.5, 3, 3, 2.5, 2],
    
    # Between roles - broad distribution with slight young skew
    "between roles": [1.5, 2, 2.5, 2, 1.5, 1, 0.8, 0.6, 0.4],
}

# Life facet categories for rich background generation
LIFE_FACET_CATEGORIES: List[str] = [
    # Identity / basic context
    "identity_stage",
    "cultural_background_orientation",
    "sense_of_belonging",
    "values_and_priorities",
    "self_view",
    
    # Goals and direction
    "short_term_goal",
    "long_term_goal_or_dream",
    "stalled_goal",
    "source_of_motivation",
    
    # Important people
    "key_partner_or_love_interest",
    "closest_friend_or_confidant",
    "family_relationship_pattern",
    "work_or_school_ally",
    "conflictual_relationship",
    
    # Work / roles / daily responsibilities
    "work_or_study_pressure",
    "sense_of_achievement",
    "role_conflicts",
    "financial_pressure_or_stability",
    "schedule_and_time_pressure",
    "responsibility_load",
    
    # Health and mental health
    "physical_health_constraints",
    "sleep_pattern_tendency",
    "existing_diagnoses_or_labels",
    "past_help_seeking",
    "body_image_concerns_or_comfort",
    
    # Stressors and vulnerabilities
    "current_primary_stressor",
    "secondary_stressors",
    "loss_or_change",
    "unresolved_issue",
    "fear_or_worry_theme",
    
    # Coping and habits
    "coping_style",
    "day_to_day_routines",
    "soothing_activities",
    "less_helpful_coping",
    "digital_or_social_media_habits",
    
    # Interests / quirks / color
    "hobbies_and_interests",
    "small_joys",
    "personal_quirks",
    "self_presentation_style",
    "areas_of_competence_or_pride",
    
    # History / adversity
    "past_difficult_period",
    "prior_relationship_disappointment_or_breakdown",
    "earlier_school_or_work_challenge",
    "family_history_of_health_or_mental_health_issues",
    "significant_move_or_transition",
    
    # Constraints and environment
    "housing_and_neighbourhood_feel",
    "access_to_resources",
    "time_and_energy_constraints",
    
    # Beliefs and meaning
    "explanatory_style",
    "beliefs_about_help_and_treatment",
    "beliefs_about_self_worth",
    "hopes_for_future",
    
    # Obsessions / preoccupations
    "preoccupations_or_obsessions",
]

# Trauma/adversity facet categories (subset of LIFE_FACET_CATEGORIES)
TRAUMA_ADVERSITY_FACETS: List[str] = [
    "past_difficult_period",
    "prior_relationship_disappointment_or_breakdown",
    "family_history_of_health_or_mental_health_issues",
    "significant_move_or_transition",
    "loss_or_change",
]
