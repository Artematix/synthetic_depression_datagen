"""
DSM-5 depression screening symptom keys.
"""
from typing import List

DSM_ITEMS: List[str] = [
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

# Legacy name for backwards compatibility during transition
QUESTIONNAIRE = [(key, "") for key in DSM_ITEMS]
