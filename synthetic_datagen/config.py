# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log level constants
LOG_MINIMAL = "minimal"  # Just dialogue (doctor/patient responses)
LOG_LIGHT = "light"      # Default. Adds manager guidance, next action, disclosure state
LOG_HEAVY = "heavy"      # Everything: tone tags, key points, emotional state, full instructions

# Available log levels for CLI validation
LOG_LEVELS = [LOG_MINIMAL, LOG_LIGHT, LOG_HEAVY]


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Random seed handling
RANDOM_SEED = None  # Default None, can be set via CLI

# OpenAI model configuration
OPENAI_MODEL = "gpt-4.1-mini"

# Available models
AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-5-mini",
]

# Default temperature and token settings
# Note: gpt-5-mini requires temperature=1.0 (no other values supported)
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 1000

# Randomization flags
RANDOM_TEMPERATURE = False
RANDOM_MAX_TOKENS = False


# ============================================================================
# AGENT-SPECIFIC SETTINGS
# ============================================================================

# Manager agent settings
MANAGER_TEMPERATURE = 1.0
MANAGER_MAX_TOKENS = 1000

# Background writer settings
BACKGROUND_WRITER_TEMPERATURE = 1.0
BACKGROUND_WRITER_MAX_TOKENS = 1000


# ============================================================================
# PACING & TURN BUDGET
# ============================================================================

# Target turns per DSM symptom based on microstyle pacing
PACING_TARGETS = {
    "brisk": 1.5,
    "med": 2.0,
    "slow": 2.5,
}

# Personas that get extended pacing (+0.5 turns per symptom, max 3.0)
EXTENDED_PACING_PERSONAS = {"very_warm_chatty", "trauma_informed_slow"}

# Number of DSM items to screen
NUM_DSM_ITEMS = 9

# Buffer turns for force-DSM guard (allows natural wrap-up after DSM coverage)
BUFFER_TURNS = 5


# ============================================================================
# SESSION DEFAULTS
# ============================================================================

# Default number of sessions to run
DEFAULT_NUM_SESSIONS = 10

# Default doctor persona for deterministic runs
DEFAULT_DOCTOR_PERSONA_ID = "warm_validating"

# Default log level
DEFAULT_LOG_LEVEL = LOG_MINIMAL
