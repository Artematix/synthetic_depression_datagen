"""
synthetic_depression_datagen_v1.py
──────────────────────────────────
Generates synthetic doctor‑patient dialogues for depression‑screening research.

High‑level pipeline
───────────────────
1.  **Patient profile generator**  
    • Samples demographics, personality traits, depression‑symptom frequencies  
    • Optionally injects random recent‑life events  
    • Renders a *system prompt* for the “patient” agent

2.  **Doctor agent**  
    • Guided by a fixed system prompt plus a rotating DSM‑5 question queue  
    • Greets → listens → asks exactly one questionnaire item each turn

3.  **Dialogue loop** (`NUM_TURNS` exchanges)  
    • Patient reply → Doctor reply, both via OpenAI chat completions  
    • Temperature and `max_tokens` may be randomised per run for diversity

4.  **Persistence layer**  
    • Saves a full JSON transcript containing:  
      ‑ agent ground‑truth profile  
      ‑ ordered questionnaire keys  
      ‑ OpenAI parameters used  
      ‑ time stamp and unique `agent_id`

File layout
───────────
• Section 0  Global constants & tunables  
• Section 1  Data pools (demographics, traits, events)  
• Section 2  Questionnaire list (9 DSM‑5 items)  
• Section 3  Doctor system prompt & helper  
• Section 4  Patient system‑prompt builder  
• Section 5  `chat_completion` wrapper (OpenAI SDK v1.0 style)  
• Section 6  Main `run_patient_doctor_session` pipeline  
• Section 7  CLI test‑hook (`__main__`)
"""

import random, json, uuid, datetime, hashlib
from openai import OpenAI                   # pip install openai
from typing import List, Dict, Any, Tuple

######################################################################
# 0. Global configuration & placeholder data stores
######################################################################

API_KEY: str                = "PLACEHOLDER"              # contact Artemy Gavrilov for API key, or generate your own at https://openai.com/api
client: OpenAI = OpenAI(api_key=API_KEY)

# Optional parameters for chat completion, not super important
TOP_P: float                = 1.0
FREQ_PENALTY: float         = 0.0
PRES_PENALTY: float         = 0.0
STOP_TOKENS: List[str]      = []
STREAM_RESPONSES: bool      = False

# Configurable Constants
OPENAI_MODEL: str           = "gpt-4o"
NUM_TURNS: int             = 14  # Number of turns in the conversation

RANDOM_TEMPERATURE: bool = False
RANDOM_MAX_TOKENS: bool = False

# If random temp or max tokens are False:
DEFAULT_TEMPERATURE: float = 1.6
DEFAULT_MAX_TOKENS: int = 200

MAX_LIFE_EVENTS: int = 2  # Max number of random life events to include in the patient profile

############################################################################
# 1.  DEMOGRAPHIC / TRAIT / EVENT DATA POOLS
############################################################################

# Demographic option pools (examples)
AGE_OPTIONS:                 List[str] = ["15-18", "18-25", "26-35", "36-45", "46-55", "56+"]
RACE_OPTIONS:                List[str] = ["White", "Black", "Asian", "Hispanic", "Other"]
GENDER_OPTIONS:              List[str] = ["Male", "Female", "Non-binary", "Other"]
SOCIOECONOMIC_OPTIONS:       List[str] = ["Low", "Medium", "High"]
COUNTRY_OF_ORIGIN_OPTIONS:   List[str] = ["USA", "Canada", "UK", "Australia", "Other"]
ENGLISH_FLUENCY_OPTIONS:     List[str] = ["Native", "Second-language", "Other"]
EDUCATION_LEVEL_OPTIONS:     List[str] = ["High School", "Some College", "Bachelor's", "Master's", "Doctorate"]

# Personality–trait buckets (examples)
TRAITS_A_COMM_STYLE: List[str] = ["Open", "Closed-off", "Distrusting", "Friendly", "Reserved"]
TRAITS_B_EMOTION:    List[str] = ["Shy", "Outgoing", "Angry", "Calm", "Anxious"]
TRAITS_C_LANGUAGE:   List[str] = ["Formal", "Casual", "Slang-heavy", "Technical", "Simple"]
TRAITS_D_LANGUAGE:   List[str] = ["Verbose", "Concise", "Descriptive", "Direct", "Indirect"]

# DSM‑5 Depression symptom catalogue (add detail for better representation by the model)
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


# Frequency categories for symptom presence across last 14 days (no idea what it should be)
FREQUENCY_CATEGORIES: Dict[str, str] = {
    "NONE": "Not at all",
    "RARE": "One or two days",
    "SOME": "Three to five days",
    "OFTEN": "Six to ten days",
}

# Random events that patients might have experienced recently (not sure if this is needed but could add more diversity to generated conversations if enough examples are found)
RANDOM_EVENTS: List[str] = [
    "Had a fight with a friend",
    "Lost a job",
    "Moved to a new city",
    "Started a new relationship",
    "Experienced a death in the family",
    "Graduated from school",
    "Went on a vacation",
    "Had a health scare",
    "Death of a pet",
    "Financial difficulties",
    "Broke up with a partner",
    "Had a car accident",
    "Received bad news from a doctor",
    "Had a major argument with a family member",
    "Lost a significant amount of money",
    "Experienced a natural disaster",   
]


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
# 3.  DOCTOR AGENT TEMPLATE (Should be tuned to our needs, can also be varied for more diverse data)
############################################################################
DOCTOR_FIRST_GREETING: str = "Hello! It’s nice to see you today. How have you been feeling lately?"

DOCTOR_SYSTEM_PROMPT: str = """
You are a primary‑care physician conducting a routine
check‑up. Workflow rules you must follow:
1. Begin by greeting the patient warmly and asking how they are.
2. After every patient reply, respond if needed, AND embed EXACTLY one
   question from the provided Questionnaire list.  Ask the questions in
   an order supplied to you (they will be given as <NEXT_QUESTION>).
"""

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
    demographics: Dict[str, str],
    personality: Dict[str, str],
    depression_profile: Dict[str, str],
    random_events: List[str] | None = None,
    extra_instruction: str = (
        "You are a patient in a doctor's office, getting a routine check‑up. "
        "Answer the doctor’s questions according to your personality and do NOT ask your own questions."
    ),
) -> str:
    """
    Returns a fully‑formed system prompt describing the patient/agent.

    Parameters
    ----------
    demographics : mapping of demographic keys to chosen option strings
        
    personality : mapping for TRAITS_A/B/C (communication style, emotion, etc.)
    
    depression_profile : mapping of each DSM‑5 symptom (string) to a frequency code in FREQUENCY_CATEGORIES

    random_events : Extra events that the patient might have experienced recently
    
    extra_instruction : any additional task‑specific meta‑instruction
    
    Keys should map to the option pools above (AGE, RACE, etc.).
    """
    
    # -- Assemble demographic blurb ------------------------------------
    demo_lines = [
        f"{key.replace('_', ' ').title()}: {val}"
        for key, val in demographics.items()
        if val
    ]

    # -- Assemble personality blurb ------------------------------------
    pers_lines = [
        f"{bucket.replace('_', ' ').title()}: {trait}"
        for bucket, trait in personality.items()
        if trait
    ]

    # -- Assemble depression timeline ----------------------------------
    dep_lines = []
    for symptom, freq_code in depression_profile.items():
        freq_verbose = FREQUENCY_CATEGORIES.get(freq_code, "N/A")
        dep_lines.append(f"{symptom} – {freq_verbose}")
        
        
    # -- Add random event (or events) (randomly selected) --------------------------
    event_lines = []
    if random_events:
        num_events = random.randint(0, MAX_LIFE_EVENTS)  # Randomly select 0 to n events
        selected_events = random.sample(random_events, min(num_events, len(random_events)))
        event_lines.extend(f"Recent Life Event: {event}" for event in selected_events)


    # -- Glue everything into a single system prompt -------------------
    prompt_sections = [
        "### Patient Demographics ###",
        *demo_lines,
        "",
        "### Personality ###",
        *pers_lines,
        "",
        "### DSM‑5 Depression Symptom Summary (past 14 days) ###",
        *dep_lines,
        "",
        "### Recent Life Events ###",
        *event_lines,
        "",
        "### Context ###",
        extra_instruction,
    ]
    return "\n".join(prompt_sections)

# -------------------------------------------------------------------
# 5. Core – generic, configurable OpenAI chat wrapper  (SDK ≥ 1.0.0)
# -------------------------------------------------------------------
def chat_completion(
    *,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    model: str = OPENAI_MODEL,
    top_p: float = TOP_P,
    frequency_penalty: float = FREQ_PENALTY,
    presence_penalty: float = PRES_PENALTY,
    stop: List[str] | None = STOP_TOKENS,
    stream: bool = STREAM_RESPONSES,
) -> str:
    """
    Wrapper around `client.chat.completions.create` that works with
    openai‑python ≥ 1.0.0.  Returns **assistant text only**.
    """
    # Build the call‑arguments dictionary once
    req_kwargs: Dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop or None,
        stream=stream,
    )

    # Non‑streaming: single call, single response object
    if not stream:
        resp = client.chat.completions.create(**req_kwargs)
        return resp.choices[0].message.content

    # Streaming: concatenate content chunks on the fly
    full_reply_chunks: List[str] = []
    for chunk in client.chat.completions.create(**req_kwargs):
        if chunk.choices[0].delta and chunk.choices[0].delta.content:
            full_reply_chunks.append(chunk.choices[0].delta.content)
    return "".join(full_reply_chunks)


############################################################################
# 6.  MAIN PIPELINE
############################################################################
def run_patient_doctor_session(
    *,
    use_random_agent: bool = False,  # True to use random agent profile, False to use a specific one
    forced_agent_profile: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Orchestrates one complete doctor–patient dialogue and
    returns a JSON‑serialisable dict capturing all dialogue and template information.
    """

    # 6‑A -> Agent Selection ----------------------------------------------
    # Determine demographics/personality/profile (random or forced)
    if use_random_agent or forced_agent_profile is None:
        demographics = {
            "AGE":                  random.choice(AGE_OPTIONS),
            "RACE":                 random.choice(RACE_OPTIONS),
            "GENDER":               random.choice(GENDER_OPTIONS),
            "SOCIOECONOMIC_STATUS": random.choice(SOCIOECONOMIC_OPTIONS),
            "COUNTRY_OF_ORIGIN":    random.choice(COUNTRY_OF_ORIGIN_OPTIONS),
            "ENGLISH_PROFICIENCY":  random.choice(ENGLISH_FLUENCY_OPTIONS),
            "EDUCATION_LEVEL":      random.choice(EDUCATION_LEVEL_OPTIONS),
        }
        personality = {
            "TRAITS_A_COMM_STYLE": random.choice(TRAITS_A_COMM_STYLE),
            "TRAITS_B_EMOTION":    random.choice(TRAITS_B_EMOTION),
            "TRAITS_C_LANGUAGE":   random.choice(TRAITS_C_LANGUAGE),
            "TRAITS_D_LANGUAGE":   random.choice(TRAITS_D_LANGUAGE),
        }
        depression_profile = {
            sym: random.choice(list(FREQUENCY_CATEGORIES.keys()))
            for sym in DSM5_DEPRESSION_SYMPTOMS
        }
        random_events = random.sample(
            RANDOM_EVENTS,
            random.randint(0, MAX_LIFE_EVENTS)
        )
    else:
        # ← allow grid‑search injection of fully‑specified profile
        demographics       = forced_agent_profile["demographics"]
        personality        = forced_agent_profile["personality"]
        depression_profile = forced_agent_profile["depression_profile"]
        random_events      = forced_agent_profile["random_events"]

    patient_system_prompt = build_patient_system_prompt(
        demographics=demographics,
        personality=personality,
        depression_profile=depression_profile,
        random_events=random_events,
    )

    # Build a reproducible key from the agent’s profile and hash it
    profile_items: List[str] = []
    for k, v in sorted(demographics.items()):
        profile_items.append(f"{k}={v}")
    for k, v in sorted(personality.items()):
        profile_items.append(f"{k}={v}")
    for symptom, freq in sorted(depression_profile.items()):
        profile_items.append(f"{symptom}={freq}")
    if random_events:
        evs = ",".join(sorted(random_events))
        profile_items.append(f"EVENTS={evs}")

    profile_key = "|".join(profile_items)
    agent_hash  = hashlib.sha256(profile_key.encode("utf-8")).hexdigest()[:16]
    agent_id    = f"AGENT_{agent_hash}"

    # 6‑B -> Random OpenAI Parameters (max tokens and temperature) -------------------
    if RANDOM_TEMPERATURE:
        temperature = round(random.uniform(0.2, 1.0), 2)
    else:
        temperature = DEFAULT_TEMPERATURE
    
    if RANDOM_MAX_TOKENS:
        max_tokens = random.randint(50, 300)
    else:
        max_tokens = DEFAULT_MAX_TOKENS

    # 6‑C -> Conversation Loop ----------------------------------------
    conversation_history: List[Dict[str, str]] = []
    questionnaire_pool: List[Tuple[str, str]] = QUESTIONNAIRE.copy()
    asked_question_order: List[str] = []

    # Doctor’s first greeting
    conversation_history.append({"role": "assistant", "content": DOCTOR_FIRST_GREETING})
    print("Doctor's first greeting:", DOCTOR_FIRST_GREETING)

    for turn_idx in range(NUM_TURNS):
        # --- Patient replies ------------------------------
        patient_messages = [
            {"role": "system", "content": patient_system_prompt},
            *conversation_history,
        ]
        patient_reply = chat_completion(
            messages=patient_messages,
            model=OPENAI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=TOP_P,
            frequency_penalty=FREQ_PENALTY,
            presence_penalty=PRES_PENALTY,
            stop=STOP_TOKENS,
            stream=STREAM_RESPONSES,
        )
        conversation_history.append({"role": "user", "content": patient_reply})
        print(f"\nPatient reply (turn {turn_idx + 1}):", patient_reply)

        # --- Doctor chooses next question -----------------
        if questionnaire_pool:
            symptom_key, next_q = random.choice(questionnaire_pool)
            questionnaire_pool.remove((symptom_key, next_q))
            asked_question_order.append(symptom_key)
        else:
            final_msg = "Thank you for sharing. Is there anything else you’d like to discuss today?"
            conversation_history.append({"role": "assistant", "content": final_msg})
            break

        doctor_messages = build_doctor_messages(
            conversation_history=conversation_history,
            next_question=next_q,
        )
        doctor_reply = chat_completion(
            messages=doctor_messages,
            model=OPENAI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=TOP_P,
            frequency_penalty=FREQ_PENALTY,
            presence_penalty=PRES_PENALTY,
            stop=STOP_TOKENS,
            stream=STREAM_RESPONSES,
        )
        conversation_history.append({"role": "assistant", "content": doctor_reply})
        print(f"\nDoctor reply (turn {turn_idx + 1}):", doctor_reply)

    # 6‑D -> Save transcript ----------------------------------------
    session_data = {
        "run_timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "agent_id":          agent_id,
        "demographics":      demographics,
        "personality":       personality,
        "depression_profile_ground_truth": depression_profile,
        "asked_question_order": asked_question_order,
        "openai_call_parameters": {
            "model":       OPENAI_MODEL,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "top_p":       TOP_P,
            "frequency_penalty": FREQ_PENALTY,
            "presence_penalty":  PRES_PENALTY,
        },
        "conversation": conversation_history,
    }

    # Persist locally (filename embeds agent_id)
    fname = f"transcript_{agent_id}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)

    return session_data   # also return for immediate inspection

############################################################################
# 7.  QUICK TEST HOOK
############################################################################
if __name__ == "__main__":
    random.seed(42)
    transcript = run_patient_doctor_session(use_random_agent=True)
    print("Saved conversation with ID:", transcript["agent_id"])