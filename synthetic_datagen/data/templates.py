"""
DSM-5 symptom lists, frequency categories, and Big Five depression templates.
"""
from typing import List, Dict, Any

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

# Frequency categories for symptom presence across last 14 days
FREQUENCY_CATEGORIES: Dict[str, str] = {
    "NONE": "Not at all",
    "RARE": "One or two days",
    "SOME": "Three to five days",
    "OFTEN": "Six to ten days",
}

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
