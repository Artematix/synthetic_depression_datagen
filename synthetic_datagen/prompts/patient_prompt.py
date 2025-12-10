"""
Patient system prompt builder.
"""
from typing import Dict, Optional
from synthetic_datagen.data.templates import FREQUENCY_CATEGORIES
from synthetic_datagen.data.life_background import PatientLifeBackground


def build_patient_system_prompt(
    *,
    personality: Dict[str, str],
    depression_profile: Dict[str, str],
    life_background: Optional[PatientLifeBackground] = None,
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
    
    life_background : PatientLifeBackground, optional
        Rich life background from background writer, if available
    """
    
    # -- Get age range (from personality dict) -------------------------
    age_range = personality.get('AGE_RANGE', 'adult')
    
    # -- Assemble Big Five Depression Template blurb ------------------
    template_lines = [
        f"**Age Range**: {age_range}",
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
        pacing_instruction = "- Provide answers with OCCASIONAL CONTEXT when relevant."


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
    
    # Add life background if present (from background writer)
    if life_background:
        # Add name and basic identity
        prompt_sections += [
            f"### Your Identity ###",
            f"**Name**: {life_background.name}",
            f"**Age**: {life_background.age_range}",
            f"**Pronouns**: {life_background.pronouns}",
            "",
        ]
        
        # Add core roles
        if life_background.core_roles:
            roles_str = ", ".join(life_background.core_roles)
            prompt_sections += [
                f"**Main roles**: {roles_str}",
                "",
            ]
        
        # Add core relationships
        if life_background.core_relationships:
            prompt_sections += [
                "**Key relationships**:",
                *[f"• {rel}" for rel in life_background.core_relationships],
                "",
            ]
        
        # Add core stressor summary
        if life_background.core_stressor_summary:
            prompt_sections += [
                f"**Current main stressors**: {life_background.core_stressor_summary}",
                "",
            ]
        
        # Add selected high-salience life facets (2-4)
        high_sal_facets = [f for f in life_background.life_facets if f.salience == "high"]
        med_sal_facets = [f for f in life_background.life_facets if f.salience == "med"]
        
        # Select 2-4 facets to include, prioritizing high salience
        facets_to_include = high_sal_facets[:2]  # Take up to 2 high-salience
        if len(facets_to_include) < 4 and med_sal_facets:
            # Fill up to 4 total with med-salience
            facets_to_include.extend(med_sal_facets[:4 - len(facets_to_include)])
        
        if facets_to_include:
            prompt_sections += [
                "### Key Life Details ###",
                "These are specific aspects of your life that may come up naturally in conversation:",
            ]
            for facet in facets_to_include:
                prompt_sections.append(f"• {facet.description}")
            prompt_sections.append("")
    
    elif bg_lines:
        # Fallback to basic personal background if no life_background
        prompt_sections += [
            "### Personal Background ###",
            "Use these as light context anchors; don't invent a detailed backstory:",
            *bg_lines,
            "",
        ]
    
    prompt_sections += [
        "### Roleplay Instructions ###",
        "",
        "Be this patient—speak with their voice, mannerisms, and personality. Let your template, modifiers, and voice style shape everything: word choice, sentence length, how much you share, and how guarded or open you are.",
        "",
        "GUIDELINES:",
        "- Answer the doctor's questions; do not ask your own",
        "- Show your feelings through your words—let emotion come through in what you say and how you say it",
        "- Draw on your life context and background naturally",
        pacing_instruction,
        "- Vary your wording—avoid using the same phrases repeatedly",
        "- If humor fits your personality, use it to cope or deflect",
        "- Output ONLY your spoken words—no stage directions, parentheticals, or action descriptions",
        "- Let the conversation tell your story—connect responses to what came before",
    ]
    return "\n".join(prompt_sections)
