import re

def _clean_number_string(number_str: str) -> float:
    # Remove commas or other thousand separators
    cleaned_str = number_str.replace(',', '')
    return float(cleaned_str)

def is_correct_answer(agent_answer: str, correct_answer: str) -> bool:
    """
    Check if the agent's answer is correct by extracting the last numerical value.
    
    Args:
        agent_answer: The agent's answer (string containing at least one numerical value)
        correct_answer: The correct answer (string that can be converted to float)
        
    Returns:
        True if the last numerical value in the agent's answer matches the correct answer, False otherwise
    """
    # Pattern to match different number formats:
    # - Integers with optional commas: -?[\d,]+
    # - Numbers with decimal points: -?[\d,]+\.?\d*
    # - Numbers starting with decimal: -?\.\d+
    # - Scientific notation: (?:[eE][-+]?\d+)?
    pattern = r"(-?[\d,]+\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+(?:[eE][-+]?\d+)?)"
    matches = re.findall(pattern, agent_answer)
    
    # Filter out invalid matches by attempting conversion to float
    valid_numbers = []
    for match in matches:
        try:
            valid_numbers.append(_clean_number_string(match))
        except (ValueError, TypeError):
            continue
    
    if not valid_numbers:
        return False
    
    try:
        # Get the last valid numerical value
        agent_numerical = valid_numbers[-1]
        correct_numerical = _clean_number_string(str(correct_answer))
        
        # Compare with small tolerance for floating point precision issues
        return abs(agent_numerical - correct_numerical) < 1e-6
    
    except (ValueError, TypeError) as e:
        print(f"Error extracting numerical values: {e}")
        return False