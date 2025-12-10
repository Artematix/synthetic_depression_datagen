"""
I/O utilities for saving session transcripts.
"""
import json
import os
from typing import Dict, Any


def save_transcript(session_data: Dict[str, Any], output_dir: str = "outputs/transcripts") -> str:
    """
    Save session transcript to JSON file.
    
    Parameters
    ----------
    session_data : dict
        Session data including conversation, metadata, and ground truth
    output_dir : str
        Directory to save transcripts (default: outputs/transcripts)
        
    Returns
    -------
    str
        Full path to saved transcript file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build filename
    agent_id = session_data.get("agent_id", "UNKNOWN")
    filename = f"transcript_{agent_id}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Write transcript
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)
    
    return filepath
