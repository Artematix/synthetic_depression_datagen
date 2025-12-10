"""
synthetic_depression_datagen_v4.py
──────────────────────────────────
Generates synthetic doctor‑patient dialogues for depression‑screening research.
Uses Big Five personality templates to drive depression presentation patterns.

High‑level pipeline
───────────────────
1.  **Patient profile generator (Template-Based)**  
    • Selects one of 8 Big Five depression templates
    • Derives personality traits and symptom frequencies from template
    • Samples modifiers, intensity levels, episode density, voice style
    • Renders a *system prompt* for the "patient" agent

2.  **Doctor agent (Persona-Based)**  
    • Selects one of multiple doctor personas
    • Guided by persona-specific system prompt + DSM‑5 question queue  
    • Uses persona-specific reflections for natural flow

3.  **Dialogue loop** (`NUM_TURNS` exchanges)  
    • Patient reply → Doctor reply (with optional reflections)
    • Both via OpenAI chat completions  

4.  **Persistence layer**  
    • Saves transcript with all metadata

File layout
───────────
• Section 0  Global constants & tunables  
• Section 1  Data pools (templates, modifiers, voice styles, background)
• Section 2  Questionnaire list (9 DSM‑5 items)  
• Section 3  Doctor personas & prompts
• Section 4  Patient system‑prompt builder  
• Section 5  Agents SDK helper constructors
• Section 6  Main `run_patient_doctor_session` pipeline  
• Section 7  CLI test‑hook (`__main__`)
"""

import random, json, uuid, datetime, hashlib, time
from openai import OpenAI, OpenAIError              # pip install openai
from agents import Agent, ModelSettings, RunConfig, Runner, SQLiteSession, set_default_openai_key                     # pip install openai-agents
from typing import List, Dict, Any, Tuple

######################################################################
# 0. Global configuration & placeholder data stores
######################################################################

# RANDOM_SEED: int = 115       # REMEMBER TO CHANGE THIS FOR EACH RUN
# Or generate it randomly
RANDOM_SEED: int = random.randint(1, 1000000)

API_KEY: str                = "PLACEHOLDER"              # contact Artemy Gavrilov for API key, or generate your own at https://openai.com/api
client: OpenAI = OpenAI(api_key=API_KEY)
runner = Runner()
set_default_openai_key(API_KEY)
session = SQLiteSession(":memory:")  # In‑memory SQLite session for stateful agents

# Optional parameters for chat completion
TOP_P: float                = 1.0
FREQ_PENALTY: float         = 0.0
PRES_PENALTY: float         = 0.0
STOP_TOKENS: List[str]      = []
STREAM_RESPONSES: bool      = False

# Configurable Constants
OPENAI_MODEL: str           = "gpt-4o"
NUM_TURNS: int             = 9  # Number of turns in the conversation

RANDOM_TEMPERATURE: bool = False
RANDOM_MAX_TOKENS: bool = False

# If random temp or max tokens are False:
DEFAULT_TEMPERATURE: float = 1.3
DEFAULT_MAX_TOKENS: int = 1000

# Reflection probability in doctor responses
REFLECTION_PROBABILITY: float = 0.4

############################################################################
# 1.  DATA POOLS
############################################################################

# DSM‑5 Depression symptom catalogue
DSM5_DEPRESSION_SYMPTOMS: List[str] = [
    "Depressed mood",
    "Loss of interest or pleasure",
    "Significant weight/appetite changes",
    "Sleep disturbances",
    "Psychomotor agitation or retardation",
    "Fatigue or loss of energy",
    "Feelings of worthlessness or excessive guilt",
    "Difficulty concentrating or indecisiveness",
    "Recurrent thoughts of death or suicide",
]

# Big Five Depression Templates - structured mapping of personality traits to depression presentations
BIG5_DEP_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "NEUROTICISM_HIGH": {
        "affective": "Intense sadness, guilt, irritability, emotional volatility",
        "cognitive": "Rumination, self-blame, helplessness, catastrophic thinking",
        "somatic": "Fatigue, sleep disturbance, tension, agitation",
        "emphasized_symptoms": [
            "Depressed mood",
            "Fatigue or loss of energy",
            "Feelings of worthlessness or excessive guilt",
            "Difficulty concentrating or indecisiveness"
        ],
        "specifier": "MDD with anxious distress",
        "modifiers": ["worry-prone", "self-blaming", "threat-sensitivity", "emotionally-reactive", "ruminative"]
    },
    "EXTRAVERSION_LOW": {
        "affective": "Emotional flatness, social withdrawal, anhedonia",
        "cognitive": "Hopelessness, pessimism, low reactivity to positives",
        "somatic": "Psychomotor slowing, hypersomnia, low energy",
        "emphasized_symptoms": [
            "Loss of interest or pleasure",
            "Psychomotor agitation or retardation",
            "Fatigue or loss of energy"
        ],
        "specifier": "Persistent depressive disorder-like",
        "modifiers": ["socially-withdrawn", "pleasure-unresponsive", "passive", "low-initiative", "isolating"]
    },
    "CONSCIENTIOUSNESS_HIGH": {
        "affective": "Controlled/suppressed affect, tension under responsibility",
        "cognitive": "Perfectionism, strong self-criticism, guilt over small failures, indecision",
        "somatic": "Insomnia, appetite loss, exhaustion from overwork",
        "emphasized_symptoms": [
            "Significant weight/appetite changes",
            "Sleep disturbances",
            "Fatigue or loss of energy",
            "Feelings of worthlessness or excessive guilt"
        ],
        "specifier": "Melancholic features",
        "modifiers": ["perfectionistic", "rigidly-self-critical", "duty-focused", "overwork-prone", "failure-intolerant"]
    },
    "CONSCIENTIOUSNESS_LOW": {
        "affective": "Apathy, disengagement, blunted emotion",
        "cognitive": "Disorganization, inefficiency, forgetfulness",
        "somatic": "Hypersomnia, low motivation, poor self-care, variable appetite",
        "emphasized_symptoms": [
            "Psychomotor agitation or retardation",
            "Fatigue or loss of energy",
            "Difficulty concentrating or indecisiveness"
        ],
        "specifier": "With functional impairment",
        "modifiers": ["disorganized", "unmotivated", "self-care-neglecting", "task-avoidant", "forgetful"]
    },
    "AGREEABLENESS_HIGH": {
        "affective": "Empathic sadness, guilt about others, over-concern",
        "cognitive": "Moral/relational rumination about failing people",
        "somatic": "Fatigue from overextending, sleep disturbance from worry",
        "emphasized_symptoms": [
            "Depressed mood",
            "Sleep disturbances",
            "Fatigue or loss of energy",
            "Feelings of worthlessness or excessive guilt"
        ],
        "specifier": "Anxious distress",
        "modifiers": ["people-pleasing", "over-responsible-for-others", "self-sacrificing", "conflict-avoidant", "guilt-prone"]
    },
    "AGREEABLENESS_LOW": {
        "affective": "Irritability, anger, frustration, externalized blame",
        "cognitive": "Defensive/hostile thoughts, rejection sensitivity",
        "somatic": "Restlessness, agitation, appetite disturbance, insomnia",
        "emphasized_symptoms": [
            "Depressed mood",
            "Psychomotor agitation or retardation",
            "Sleep disturbances"
        ],
        "specifier": "Mixed features",
        "modifiers": ["irritable", "blame-externalizing", "rejection-sensitive", "defensively-hostile", "interpersonally-strained"]
    },
    "OPENNESS_HIGH": {
        "affective": "Existential sadness, metaphorical expression of distress",
        "cognitive": "Philosophical rumination on meaning and mortality",
        "somatic": "Variable; fatigue from over-reflection",
        "emphasized_symptoms": [
            "Depressed mood",
            "Loss of interest or pleasure",
            "Recurrent thoughts of death or suicide"
        ],
        "specifier": "Mild MDD / adjustment-like",
        "modifiers": ["existentially-focused", "meaning-seeking", "introspective", "metaphorically-expressive", "philosophically-ruminating"]
    },
    "OPENNESS_LOW": {
        "affective": "Constricted affect, limited emotional vocabulary",
        "cognitive": "Literal thinking, low insight, denies mood distress",
        "somatic": "Body-focused complaints (aches, tiredness, sleep/appetite issues)",
        "emphasized_symptoms": [
            "Significant weight/appetite changes",
            "Sleep disturbances",
            "Fatigue or loss of energy"
        ],
        "specifier": "Somatic-dominant style",
        "modifiers": ["somatically-focused", "insight-limited", "mood-distress-denying", "literal-thinking", "body-complaint-oriented"]
    }
}

# Frequency categories for symptom presence across last 14 days
FREQUENCY_CATEGORIES: Dict[str, str] = {
    "NONE": "Not at all",
    "RARE": "One or two days",
    "SOME": "Three to five days",
    "OFTEN": "Six to ten days",
}

# Conversation pacing levels (patient elaboration style)
PACING_LEVELS: List[str] = ["LOW", "MED", "HIGH"]

# Generic context domains for life situation framing
GENERIC_CONTEXT_DOMAINS: List[str] = [
    "work/role strain",
    "relationships strain",
    "health concern",
    "self-worth/identity strain",
    "general stress/no clear trigger"
]
MAX_CONTEXT_DOMAINS: int = 2

# Symptom intensity levels (for emphasized symptoms)
INTENSITY_LEVELS: List[str] = ["LOW", "MED", "HIGH"]

# Episode density levels (overall symptom sparsity)
EPISODE_DENSITY_LEVELS: List[str] = ["LOW", "MED", "HIGH"]

# Voice/Style profiles (verbosity, expressiveness, trust, intellect)
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


############################################################################
# 2.  QUESTIONNAIRE
############################################################################
QUESTIONNAIRE: List[Tuple[str, str]] = [
    ("Depressed mood",                                  "Over the past two weeks, how often have you felt down or hopeless?"),
    ("Loss of interest or pleasure",                    "Have you lost interest in activities you usually enjoy?"),
    ("Significant weight/appetite changes",             "Have you noticed any significant changes in your appetite or weight?"),
    ("Sleep disturbances",                              "How have your sleeping patterns been recently?"),
    ("Psychomotor agitation or retardation",            "Have you felt unusually restless or sluggish?"),
    ("Fatigue or loss of energy",                       "How often have you felt tired or without energy?"),
    ("Feelings of worthlessness or excessive guilt",    "Have you been troubled by feelings of worthlessness or guilt?"),
    ("Difficulty concentrating or indecisiveness",      "Have you had trouble concentrating or making decisions?"),
    ("Recurrent thoughts of death or suicide",          "Have you experienced thoughts about death or harming yourself?"),
]


############################################################################
# 3.  DOCTOR PERSONAS
############################################################################

# Default persona ID for deterministic runs
DEFAULT_DOCTOR_PERSONA_ID: str = "warm_validating"

# Doctor persona registry - different communication styles
DOCTOR_PERSONAS: List[Dict[str, Any]] = [
    {
        "id": "warm_validating",
        "first_greeting": "Hello! It's nice to see you today. How have you been feeling lately?",
        "system_prompt": """
You are a warm, validating primary-care physician conducting a routine check-up.
Workflow rules:
1. Greet the patient warmly and ask how they are.
2. After every patient reply, acknowledge their feelings when appropriate, then ask EXACTLY one 
   question from the provided Questionnaire. Questions will be given as <NEXT_QUESTION>.
3. Your tone should be supportive and empathetic.
""",
        "reflections": [
            "That sounds really difficult.",
            "I appreciate you sharing that with me.",
            "Thank you for being so open.",
            "I hear what you're saying.",
            "That must be challenging.",
        ]
    },
    {
        "id": "neutral_efficient",
        "first_greeting": "Good to see you. Let's talk about how you've been doing.",
        "system_prompt": """
You are a professional, efficient primary-care physician conducting a routine check-up.
Workflow rules:
1. Greet the patient and begin the assessment.
2. After every patient reply, respond briefly if needed, then ask EXACTLY one 
   question from the provided Questionnaire. Questions will be given as <NEXT_QUESTION>.
3. Your tone should be neutral, professional, and focused on gathering information.
""",
        "reflections": [
            "I see.",
            "Understood.",
            "Okay.",
            "Got it.",
            "Thank you.",
        ]
    },
    {
        "id": "gentle_brisk",
        "first_greeting": "Hi there. I'd like to check in with you about a few things today.",
        "system_prompt": """
You are a gentle but brisk primary-care physician conducting a routine check-up.
Workflow rules:
1. Greet the patient kindly and move through the assessment efficiently.
2. After every patient reply, acknowledge briefly, then ask EXACTLY one 
   question from the provided Questionnaire. Questions will be given as <NEXT_QUESTION>.
3. Your tone should be kind but direct, keeping the conversation moving.
""",
        "reflections": [
            "I understand.",
            "That's helpful to know.",
            "Alright.",
            "I appreciate that.",
            "Thanks for telling me.",
        ]
    }
]

def build_doctor_messages(
    *,
    conversation_history: List[Dict[str, str]],
    next_question: str,
) -> List[Dict[str, str]]:
    """
    Re‑materialises the message array for the doctor OpenAI call on
    every turn, ensuring the strict system prompt + history + directive.
    """
    msgs = [
        {"role": "system", "content": DOCTOR_SYSTEM_PROMPT},
        *conversation_history,
        {
            "role": "user",
            "content": f"<NEXT_QUESTION>\n{next_question}\n</NEXT_QUESTION>",
        },
    ]
    return msgs


############################################################################
# 4.  PATIENT AGENT TEMPLATE 
############################################################################
def build_patient_system_prompt(
    *,
    personality: Dict[str, str],
    depression_profile: Dict[str, str],
) -> str:
    """
    Returns a fully‑formed system prompt describing the patient/agent based on Big Five template.

    Parameters
    ----------
    personality : dict containing Big Five template information:
        - BIG5_TEMPLATE: template ID (e.g., NEUROTICISM_HIGH)
        - AFFECTIVE_STYLE: emotional presentation style
        - COGNITIVE_STYLE: thinking patterns
        - SOMATIC_STYLE: physical symptom patterns
        - SPECIFIER_HINT: DSM-5 specifier
        - PACING: conversation elaboration level (LOW/MED/HIGH)
        - CONTEXT_DOMAINS: list of life domain stressors
        - MODIFIERS: list of generic trait modifiers
        - VOICE_STYLE: dict with verbosity/expressiveness/trust/intellect
        - PERSONAL_BACKGROUND: dict with living/work/routine/support
    
    depression_profile : mapping of each DSM‑5 symptom (string) to a frequency code in FREQUENCY_CATEGORIES
    """
    
    # -- Assemble Big Five Depression Template blurb ------------------
    template_lines = [
        f"**Template Category**: {personality.get('BIG5_TEMPLATE', 'N/A')}",
        f"**Affective Style**: {personality.get('AFFECTIVE_STYLE', 'N/A')}",
        f"**Cognitive Style**: {personality.get('COGNITIVE_STYLE', 'N/A')}",
        f"**Somatic Style**: {personality.get('SOMATIC_STYLE', 'N/A')}",
        f"**Clinical Specifier**: {personality.get('SPECIFIER_HINT', 'N/A')}",
        f"**Conversation Pacing**: {personality.get('PACING', 'N/A')}",
    ]
    
    # Add voice style details
    vs = personality.get("VOICE_STYLE", {})
    template_lines += [
        f"**Verbosity**: {vs.get('verbosity', 'N/A')}",
        f"**Emotional Expressiveness**: {vs.get('expressiveness', 'N/A')}",
        f"**Trust In Doctor**: {vs.get('trust', 'N/A')}",
        f"**Intellectual Functioning**: {vs.get('intellect', 'N/A')}",
    ]

    # -- Assemble template modifiers -----------------------------------
    modifiers = personality.get('MODIFIERS', [])
    if modifiers:
        modifier_lines = [f"• {mod}" for mod in modifiers]
    else:
        modifier_lines = ["• None selected"]

    # -- Assemble depression symptom summary ---------------------------
    dep_lines = []
    for symptom, freq_code in depression_profile.items():
        freq_verbose = FREQUENCY_CATEGORIES.get(freq_code, "N/A")
        dep_lines.append(f"• {symptom} – {freq_verbose}")

    # -- Assemble context domains --------------------------------------
    context_domains = personality.get('CONTEXT_DOMAINS', [])
    if context_domains:
        domain_lines = [f"• {domain}" for domain in context_domains]
    else:
        domain_lines = ["• No clear trigger you can identify"]

    # -- Assemble personal background ----------------------------------
    background = personality.get('PERSONAL_BACKGROUND', {})
    if background:
        bg_lines = [f"• {k.replace('_', ' ').title()}: {v}" for k, v in background.items()]
    else:
        bg_lines = []

    # -- Build pacing-specific instructions ---------------------------
    pacing = personality.get('PACING', 'MED')
    if pacing == "LOW":
        pacing_instruction = "- Keep responses SHORT and MINIMAL. Only elaborate if directly asked for more details."
    elif pacing == "HIGH":
        pacing_instruction = "- Provide MORE SPONTANEOUS ELABORATION. Share context and details naturally, but still don't ask questions back."
    else:  # MED
        pacing_instruction = "- Provide 1-2 sentence answers with OCCASIONAL CONTEXT when relevant."

    # -- Voice style instructions --------------------------------------
    voice_instructions = [
        "- **Verbosity** controls response length and level of detail",
        "- **Emotional Expressiveness** controls how you express affect (flat/balanced/intense)",
        "- **Trust In Doctor** controls openness vs guardedness in your responses",
        "- **Intellectual Functioning** controls complexity and organization of your answers",
    ]

    # -- Glue everything into a single system prompt -------------------
    prompt_sections = [
        "### Big Five Depression Template ###",
        "You embody a specific depression presentation pattern based on personality traits:",
        *template_lines,
        "",
        "### Template Modifiers ###",
        "These modifiers subtly shape your tone and focus:",
        *modifier_lines,
        "",
        "### DSM‑5 Depression Symptom Profile (past 14 days) ###",
        *dep_lines,
        "",
        "### Current Life Context (Broad Domains) ###",
        *domain_lines,
        "",
    ]
    
    # Add personal background if present
    if bg_lines:
        prompt_sections += [
            "### Personal Background ###",
            "Use these as light context anchors; don't invent a detailed backstory:",
            *bg_lines,
            "",
        ]
    
    prompt_sections += [
        "### Instructions ###",
        "- You are a patient in a doctor's office, getting a routine check‑up",
        "- Simply answer the doctor's questions according to your personality and symptom profile",
        "- DO NOT ask your own questions",
        "- Express your symptoms and emotions naturally based on your affective, cognitive, and somatic styles",
        "- When relevant, frame your answers through the life domains listed above, without inventing specific events",
        "- Let the template modifiers subtly shape your tone/focus, without inventing specific events",
        pacing_instruction,
        *voice_instructions,
        "- Vary wording naturally across turns; avoid repeating the same phrases while staying consistent with your template, modifiers, and voice style",
    ]
    return "\n".join(prompt_sections)

# -------------------------------------------------------------------
# 5.  AGENTS‑SDK HELPER CONSTRUCTORS
# -------------------------------------------------------------------

def build_doctor_agent(*, instructions: str, model_settings: ModelSettings) -> Agent:
    """Return a stateful doctor agent with its persona-specific system prompt."""
    return Agent(
        name         = "SimDoctor",
        model        = OPENAI_MODEL,
        instructions = instructions,
        model_settings = model_settings,
    )


def build_patient_agent(*, patient_system_prompt: str, model_settings: ModelSettings) -> Agent:
    """Return a stateful patient agent with its personality prompt."""
    return Agent(
        name         = "SimPatient",
        model        = OPENAI_MODEL,
        instructions = patient_system_prompt,
        model_settings = model_settings,
    )



# -------------------------------------------------------------------
# 6.  MAIN PIPELINE  – Agents SDK version
# -------------------------------------------------------------------
def run_patient_doctor_session(
    *,
    use_random_agent: bool = False,
    forced_agent_profile: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Executes a multi‑turn doctor–patient dialogue with two
    stateful agents (doctor / patient) built via the Agents SDK.
    """

    # 6‑A -> Agent Profile Generation (Template-Based) ──────────────
    # Select a Big Five depression template and generate profile
    if use_random_agent or forced_agent_profile is None:
        # Select one of the 8 Big Five templates
        template_id = random.choice(list(BIG5_DEP_TEMPLATES.keys()))
        template = BIG5_DEP_TEMPLATES[template_id]
        
        # Select doctor persona
        doctor_persona = random.choice(DOCTOR_PERSONAS)
        doctor_persona_id = doctor_persona["id"]
        
        # Derive personality from template (replacing random trait sampling)
        personality = {
            "BIG5_TEMPLATE": template_id,
            "AFFECTIVE_STYLE": template["affective"],
            "COGNITIVE_STYLE": template["cognitive"],
            "SOMATIC_STYLE": template["somatic"],
            "SPECIFIER_HINT": template["specifier"],
        }
        
        # Sample template modifiers (0-2)
        modifiers = random.sample(template["modifiers"], k=random.randint(0, 2))
        personality["MODIFIERS"] = modifiers
        
        # Sample conversation pacing
        pacing = random.choice(PACING_LEVELS)
        personality["PACING"] = pacing
        
        # Sample context domains (0-2)
        num_domains = random.randint(0, MAX_CONTEXT_DOMAINS)
        context_domains = random.sample(GENERIC_CONTEXT_DOMAINS, k=num_domains)
        personality["CONTEXT_DOMAINS"] = context_domains
        
        # Sample episode density
        episode_density = random.choices(
            EPISODE_DENSITY_LEVELS, weights=[3, 5, 2], k=1
        )[0]
        personality["EPISODE_DENSITY"] = episode_density
        
        # Sample voice style
        voice_style = random.choice(VOICE_STYLE_PROFILES)
        personality["VOICE_STYLE"] = voice_style
        
        # Sample personal background (2-3 facts)
        personal_background = {
            "living_situation": random.choice(LIVING_SITUATION_POOL),
            "work_role": random.choice(WORK_ROLE_POOL),
            "routine_stability": random.choice(ROUTINE_STABILITY_POOL),
            "support_level": random.choice(SUPPORT_LEVEL_POOL),
        }
        # Drop 1-2 fields randomly so only 2-3 remain
        # keys = list(personal_background.keys())
        # for k in random.sample(keys, k=random.randint(1, 2)):
        #     personal_background.pop(k)
        # personality["PERSONAL_BACKGROUND"] = personal_background

        personality["PERSONAL_BACKGROUND"] = personal_background
        
        # Sample intensity per emphasized symptom
        emphasized = set(template["emphasized_symptoms"])
        emph_intensity = {}
        for sym in emphasized:
            emph_intensity[sym] = random.choices(
                INTENSITY_LEVELS, weights=[3, 5, 2], k=1
            )[0]
        personality["EMPH_INTENSITY"] = emph_intensity
        
        # Select 0-1 non-emphasized symptoms for extra elevation
        non_emphasized = [s for s in DSM5_DEPRESSION_SYMPTOMS if s not in emphasized]
        extra_high = random.sample(non_emphasized, k=random.randint(0, 1))
        personality["EXTRA_ELEVATED_SYMPTOMS"] = extra_high
        
        # Define density-based weights for non-emphasized symptoms
        if episode_density == "LOW":
            non_emph_weights = (7, 2, 1)  # NONE, RARE, SOME
        elif episode_density == "HIGH":
            non_emph_weights = (3, 3, 4)
        else:  # MED
            non_emph_weights = (5, 3, 2)
        
        # Generate template-biased depression profile with intensity
        depression_profile = {}
        
        for symptom in DSM5_DEPRESSION_SYMPTOMS:
            if symptom in emphasized:
                # Apply intensity-based sampling for emphasized symptoms
                level = emph_intensity[symptom]
                if level == "LOW":
                    depression_profile[symptom] = random.choices(
                        ["RARE", "SOME", "OFTEN"], weights=[5, 4, 1], k=1
                    )[0]
                elif level == "HIGH":
                    depression_profile[symptom] = random.choices(
                        ["SOME", "OFTEN"], weights=[2, 8], k=1
                    )[0]
                else:  # MED
                    depression_profile[symptom] = random.choices(
                        ["SOME", "OFTEN", "RARE"], weights=[5, 4, 1], k=1
                    )[0]
            elif symptom in extra_high:
                # Treat as secondary emphasized (lighter elevation)
                depression_profile[symptom] = random.choices(
                    ["RARE", "SOME", "OFTEN"], weights=[2, 5, 3], k=1
                )[0]
            else:
                # Non-emphasized: use density-based weights
                depression_profile[symptom] = random.choices(
                    ["NONE", "RARE", "SOME"], weights=non_emph_weights, k=1
                )[0]
    else:
        # Allow grid-search injection of fully-specified profile
        template_id        = forced_agent_profile["template_id"]
        template           = BIG5_DEP_TEMPLATES[template_id]
        personality        = forced_agent_profile["personality"]
        depression_profile = forced_agent_profile["depression_profile"]
        doctor_persona_id  = forced_agent_profile.get("doctor_persona_id", DEFAULT_DOCTOR_PERSONA_ID)
        doctor_persona     = next((p for p in DOCTOR_PERSONAS if p["id"] == doctor_persona_id), DOCTOR_PERSONAS[0])

    # Build patient system prompt with template information
    patient_system_prompt = build_patient_system_prompt(
        personality=personality,
        depression_profile=depression_profile,
    )

    # Deterministic agent_id (includes template_id for uniqueness)
    profile_key = f"TEMPLATE={template_id}|PERSONA={doctor_persona_id}|PACING={personality.get('PACING', 'MED')}|" + "|".join(
        [f"{s}={depression_profile[s]}" for s in sorted(depression_profile)]
    )
    agent_hash = hashlib.sha256(profile_key.encode()).hexdigest()[:16]
    agent_id   = f"AGENT_{agent_hash}"

    # 6‑B ▸ Session‑level generation settings ────────────────────
    patient_temperature = (
        round(random.uniform(0.2, 1.4), 2)
        if RANDOM_TEMPERATURE else DEFAULT_TEMPERATURE
    )

    doctor_temperature = (
        round(random.uniform(0.2, 1.4), 2)
        if RANDOM_TEMPERATURE else DEFAULT_TEMPERATURE
    )

    max_tokens = (
        random.randint(50, 300)
        if RANDOM_MAX_TOKENS else DEFAULT_MAX_TOKENS
    )

    run_cfg_doctor = RunConfig(
        model_settings=ModelSettings(
            temperature=doctor_temperature,
            max_tokens=max_tokens,
        )
    )

    run_cfg_patient = RunConfig(
        model_settings=ModelSettings(
            temperature=patient_temperature,
            max_tokens=max_tokens,
        )
    )
    

    # 6‑C ▸ Instantiate stateful agents ──────────────────────────
    doctor_agent = build_doctor_agent(
        instructions=doctor_persona["system_prompt"],
        model_settings=run_cfg_doctor.model_settings
    )
    patient_agent = build_patient_agent(
        patient_system_prompt=patient_system_prompt,
        model_settings=run_cfg_patient.model_settings
    )

    # 6‑D ▸ Dialogue loop ─────────────────────────────────────────
    conversation_history: List[Dict[str, str]] = []
    questionnaire_pool            = QUESTIONNAIRE.copy()
    asked_question_order: List[str] = []

    # Use persona-specific first greeting
    doctor_first_greeting = doctor_persona["first_greeting"]
    print("Doctor's first greeting:", doctor_first_greeting)
    conversation_history.append({"role": "assistant", "content": doctor_first_greeting})

    res = runner.run_sync(
        patient_agent,
        doctor_first_greeting,
        session=session,
    )
    patient_reply = res.final_output

    print("Patient reply (turn 1):", patient_reply)
    conversation_history.append({"role": "user", "content": patient_reply})

    turn = 1
    while questionnaire_pool and turn < NUM_TURNS:
        symptom_key, next_q = random.choice(questionnaire_pool)
        questionnaire_pool.remove((symptom_key, next_q))
        asked_question_order.append(symptom_key)

        # Doctor's turn - optionally prepend a reflection
        turn += 1
        
        # Sample reflection with probability
        reflection = ""
        if random.random() < REFLECTION_PROBABILITY:
            reflection = random.choice(doctor_persona["reflections"]) + "\n\n"
        
        doctor_input = f"{reflection}{patient_reply}\n<NEXT_QUESTION>\n{next_q}\n</NEXT_QUESTION>"

        dr = runner.run_sync(doctor_agent, doctor_input, session=session,)
        doctor_reply = dr.final_output

        print(f"Doctor reply (turn {turn}):", doctor_reply)
        conversation_history.append({"role": "assistant", "content": doctor_reply})

        # Patient's turn
        turn += 1
        pr = runner.run_sync(patient_agent, doctor_reply, session=session,)
        patient_reply = pr.final_output

        print(f"Patient reply (turn {turn}):", patient_reply)
        conversation_history.append({"role": "user", "content": patient_reply})

    # Close if patient spoke last
    if conversation_history[-1]["role"] == "user":
        final_msg = ("Thank you for sharing. Is there anything else "
                     "you'd like to discuss today?")
        print("Doctor final:", final_msg)
        dr = runner.run_sync(doctor_agent, final_msg, session=session,)
        conversation_history.append({"role": "assistant", "content": final_msg})

    # 6‑E ▸ Save transcript with all metadata ────────────────────
    session_data = {
        "run_timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "random_seed":      RANDOM_SEED,
        "agent_id":          agent_id,
        "big5_template_id":  template_id,
        "big5_template_details": template,
        "doctor_persona_id": doctor_persona_id,
        "doctor_persona_details": doctor_persona,
        "personality":       personality,
        "depression_profile_ground_truth": depression_profile,
        "asked_question_order": asked_question_order,
        "openai_call_parameters": {
            "model":       OPENAI_MODEL,
            "doctor_temperature": doctor_temperature,
            "patient_temperature": patient_temperature,
            "top_p":       TOP_P,
        },
        "conversation": conversation_history,
    }
    fname = f"transcript_{agent_id}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)

    return session_data



############################################################################
# 7.  QUICK TEST HOOK
############################################################################
if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    transcript = run_patient_doctor_session(use_random_agent=True)
    print("Saved conversation with ID:", transcript["agent_id"])
    print("Template used:", transcript["big5_template_id"])
    print("Doctor persona:", transcript["doctor_persona_id"])
    print("Pacing:", transcript["personality"]["PACING"])
    print("Context domains:", transcript["personality"]["CONTEXT_DOMAINS"])
