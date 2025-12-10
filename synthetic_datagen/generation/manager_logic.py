"""
Doctor manager and patient manager logic helpers for conversation management.
"""
import json
import random
from typing import Dict, List, Tuple, Any


def build_manager_input(
    conversation_history: List[Dict[str, str]],
    dsm_symptom_keys: List[str],
    doctor_persona_id: str,
    doctor_persona_style: str,
    patient_meta: Dict[str, Any],
    remaining_doctor_turns: int,
) -> str:
    """
    Build manager input with full conversation context (stateless operation).
    
    Parameters
    ----------
    conversation_history : list
        Full conversation history with role/content dicts
    dsm_symptom_keys : list
        Remaining DSM symptom keys (strings only)
    doctor_persona_id : str
        ID of the doctor persona being used
    doctor_persona_style : str
        Brief description of doctor persona style
    patient_meta : dict
        Minimal patient metadata (template_id, trust, verbosity, pacing, risk_summary, patient_background, etc.)
    remaining_doctor_turns : int
        Number of doctor turns remaining in session
        
    Returns
    -------
    str
        Formatted manager input string
    """
    # Provide full conversation history (manager is stateless)
    conversation_context = "\n".join([
        f"{'Doctor' if msg['role'] == 'assistant' else 'Patient'}: {msg['content']}"
        for msg in conversation_history
    ])
    
    # Format remaining DSM symptom keys
    remaining_dsm_list = "\n".join([f"- {key}" for key in dsm_symptom_keys])
    
    # Extract patient meta
    template_id = patient_meta.get("template_id", "N/A")
    trust = patient_meta.get("trust", "N/A")
    verbosity = patient_meta.get("verbosity", "N/A")
    pacing = patient_meta.get("pacing", "N/A")
    episode_density = patient_meta.get("episode_density", "N/A")
    modifiers = patient_meta.get("modifiers", [])
    
    # Get patient background and risk summary
    patient_background = patient_meta.get("patient_background", "N/A")
    risk_summary = patient_meta.get("risk_summary", "N/A")
    
    # Get microstyle tags
    warmth = patient_meta.get("warmth", "N/A")
    directness = patient_meta.get("directness", "N/A")
    microstyle_pacing = patient_meta.get("microstyle_pacing", "N/A")
    
    manager_input = f"""Doctor persona:
id: {doctor_persona_id}
style: {doctor_persona_style}
microstyle: warmth={warmth}, directness={directness}, pacing={microstyle_pacing}

Patient background: {patient_background}
Patient risk summary: {risk_summary}

Patient profile:
template_id: {template_id}
trust: {trust}
verbosity: {verbosity}
pacing: {pacing}
modifiers: {modifiers}
episode_density: {episode_density}

DSM symptom keys:
{remaining_dsm_list}

Conversation so far:
{conversation_context}


Task: Decide next_action and output JSON only."""
    
    return manager_input


def parse_doctor_manager_output(manager_output: str, dsm_symptom_keys: List[str]) -> Dict[str, Any]:
    """
    Parse doctor manager JSON output with fallbacks.
    
    Parameters
    ----------
    manager_output : str
        Raw output from doctor manager agent
    dsm_symptom_keys : list
        Remaining DSM symptom keys for fallback selection
        
    Returns
    -------
    dict
        Parsed decision with keys:
        - next_action: "FOLLOW_UP", "RAPPORT", or "DSM"
        - reason: str
        - doctor_instruction: str
        - dsm_symptom_key: str (empty if not DSM)
    """
    try:
        # Strip markdown code fences if present
        cleaned_output = manager_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]
        cleaned_output = cleaned_output.strip()
        
        manager_decision = json.loads(cleaned_output)
        next_action = manager_decision.get("next_action", "DSM")
        reason = manager_decision.get("reason", "")
        doctor_instruction = manager_decision.get("doctor_instruction", "") or ""
        dsm_symptom_key = manager_decision.get("dsm_symptom_key", "")
        
        # Empty doctor_instruction fallback
        if not doctor_instruction.strip():
            if next_action == "DSM":
                doctor_instruction = f"Transition naturally to asking about {dsm_symptom_key if dsm_symptom_key else 'the next symptom area'}."
            elif next_action == "FOLLOW_UP":
                doctor_instruction = "Follow up on what the patient just said."
            else:  # RAPPORT
                doctor_instruction = "Connect with the patient as a person."
        
        return {
            "next_action": next_action,
            "reason": reason,
            "doctor_instruction": doctor_instruction,
            "dsm_symptom_key": dsm_symptom_key,
        }
        
    except json.JSONDecodeError:
        # DSM fallback if invalid JSON
        if dsm_symptom_keys:
            fallback_key = random.choice(dsm_symptom_keys)
            return {
                "next_action": "DSM",
                "reason": "Doctor manager returned invalid JSON, using fallback",
                "doctor_instruction": f"Ask about {fallback_key}.",
                "dsm_symptom_key": fallback_key,
            }
        else:
            # If no DSM items left, fall back to FOLLOW_UP
            return {
                "next_action": "FOLLOW_UP",
                "reason": "Doctor manager returned invalid JSON, using follow-up fallback",
                "doctor_instruction": "Follow up on what the patient just said.",
                "dsm_symptom_key": "",
            }


def build_patient_manager_input(
    last_doctor_message: str,
    last_doctor_move: str,
    conversation_history: List[Dict[str, str]],
    patient_meta: Dict[str, Any],
    depression_profile: Dict[str, str],
    disclosure_state: str,
) -> str:
    """
    Build patient manager input with full conversation context (stateless operation).
    
    Parameters
    ----------
    last_doctor_message : str
        The most recent doctor message
    last_doctor_move : str
        Type of doctor's last action: "DSM", "FOLLOW_UP", or "RAPPORT"
    conversation_history : list
        Full conversation history with role/content dicts
    patient_meta : dict
        Patient metadata including template_id, modifiers, voice_style, etc.
    depression_profile : dict
        Symptom -> frequency mapping
    disclosure_state : str
        Current disclosure state: "MINIMIZE", "PARTIAL", or "OPEN"
        
    Returns
    -------
    str
        Formatted patient manager input string
    """
    # Provide full conversation history (manager is stateless)
    conversation_context = "\n".join([
        f"{'Doctor' if msg['role'] == 'assistant' else 'Patient'}: {msg['content']}"
        for msg in conversation_history
    ])
    
    # Format depression profile
    dep_profile_lines = "\n".join([
        f"{symptom}: {freq}" for symptom, freq in depression_profile.items()
    ])
    
    # Extract patient metadata
    template_id = patient_meta.get("template_id", "N/A")
    modifiers = patient_meta.get("modifiers", [])
    voice_style = patient_meta.get("voice_style", {})
    pacing = patient_meta.get("pacing", "N/A")
    episode_density = patient_meta.get("episode_density", "N/A")
    emphasized_symptoms = patient_meta.get("emphasized_symptoms", [])
    
    # Format voice style
    trust = voice_style.get("trust", "N/A")
    verbosity = voice_style.get("verbosity", "N/A")
    expressiveness = voice_style.get("expressiveness", "N/A")
    intellect = voice_style.get("intellect", "N/A")
    
    manager_input = f"""Patient profile:
template_id: {template_id}
voice_style: trust={trust}, verbosity={verbosity}, expressiveness={expressiveness}, intellect={intellect}
pacing: {pacing}
episode_density: {episode_density}
modifiers: {modifiers}
emphasized_symptoms: {emphasized_symptoms}

Current disclosure_state: {disclosure_state}

Depression profile (DSM symptoms):
{dep_profile_lines}

Doctor last move type: {last_doctor_move}

Full conversation so far:
{conversation_context}

Doctor last message to respond to:
{last_doctor_message}

Task:
Decide how the patient should respond next and output JSON only."""
    
    return manager_input


def parse_post_dsm_manager_output(manager_output: str) -> Dict[str, Any]:
    """
    Parse post-DSM doctor manager JSON output with fallbacks.
    
    Parameters
    ----------
    manager_output : str
        Raw output from post-DSM doctor manager agent
        
    Returns
    -------
    dict
        Parsed decision with keys:
        - next_action: "FOLLOW_UP", "RAPPORT", or "END"
        - reason: str
        - doctor_instruction: str
    """
    try:
        # Strip markdown code fences if present
        cleaned_output = manager_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]
        cleaned_output = cleaned_output.strip()
        
        manager_decision = json.loads(cleaned_output)
        next_action = manager_decision.get("next_action", "END")
        reason = manager_decision.get("reason", "")
        doctor_instruction = manager_decision.get("doctor_instruction", "") or ""
        
        # Empty doctor_instruction fallback
        if not doctor_instruction.strip():
            if next_action == "END":
                doctor_instruction = "Wrap up the visit warmly."
            elif next_action == "FOLLOW_UP":
                doctor_instruction = "Follow up on what the patient just said."
            else:  # RAPPORT
                doctor_instruction = "Respond with empathy."
        
        return {
            "next_action": next_action,
            "reason": reason,
            "doctor_instruction": doctor_instruction,
        }
        
    except json.JSONDecodeError:
        # Default to END if invalid JSON
        return {
            "next_action": "END",
            "reason": "Post-DSM manager returned invalid JSON, defaulting to END",
            "doctor_instruction": "Wrap up the visit warmly.",
        }


def parse_patient_manager_output(
    text: str,
    base_voice_style: Dict[str, str],
    current_disclosure_state: str
) -> Dict[str, Any]:
    """
    Parse patient manager JSON output with fallbacks.
    
    Parameters
    ----------
    text : str
        Raw output from patient manager agent
    base_voice_style : dict
        Voice style dict with trust, verbosity, expressiveness, intellect
    current_disclosure_state : str
        Current disclosure state for fallback
        
    Returns
    -------
    dict
        Parsed guidance with keys:
        - directness: "LOW", "MED", or "HIGH"
        - disclosure_stage: "MINIMIZE", "PARTIAL", or "OPEN"
        - target_length: "SHORT", "MEDIUM", or "LONG"
        - emotional_state: "neutral" or an emotion like "tearful", "frustrated", etc.
        - tone_tags: list of strings
        - key_points_to_reveal: list of strings
        - key_points_to_avoid: list of strings
        - patient_instruction: str
    """
    # Default values based on voice_style
    verbosity = base_voice_style.get("verbosity", "moderate")
    trust = base_voice_style.get("trust", "neutral")
    
    # Map verbosity to target_length
    if verbosity == "terse":
        default_length = "SHORT"
    elif verbosity == "detailed":
        default_length = "LONG"
    else:
        default_length = "MEDIUM"
    
    # Map trust to directness
    if trust == "guarded":
        default_directness = "LOW"
    elif trust == "open":
        default_directness = "HIGH"
    else:
        default_directness = "MED"
    
    defaults = {
        "directness": default_directness,
        "disclosure_stage": current_disclosure_state,
        "target_length": default_length,
        "emotional_state": "neutral",
        "tone_tags": ["cooperative"],
        "key_points_to_reveal": [],
        "key_points_to_avoid": [],
        "patient_instruction": "Answer in a way consistent with your profile, moderately direct, and do not overshare."
    }
    
    try:
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        parsed = json.loads(cleaned)
        
        # Fill in any missing fields with defaults
        result = {
            "directness": parsed.get("directness", defaults["directness"]),
            "disclosure_stage": parsed.get("disclosure_stage", defaults["disclosure_stage"]),
            "target_length": parsed.get("target_length", defaults["target_length"]),
            "emotional_state": parsed.get("emotional_state", defaults["emotional_state"]),
            "tone_tags": parsed.get("tone_tags", defaults["tone_tags"]),
            "key_points_to_reveal": parsed.get("key_points_to_reveal", defaults["key_points_to_reveal"]),
            "key_points_to_avoid": parsed.get("key_points_to_avoid", defaults["key_points_to_avoid"]),
            "patient_instruction": parsed.get("patient_instruction", defaults["patient_instruction"])
        }
        
        return result
        
    except (json.JSONDecodeError, ValueError):
        # Return defaults if parsing fails
        return defaults
