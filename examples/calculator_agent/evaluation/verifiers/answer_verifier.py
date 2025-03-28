import re

def _clean_number_string(number_str: str) -> float:
    # Remove commas or other thousand separators
    cleaned_str = number_str.replace(',', '')
    return float(cleaned_str)

def is_correct_answer(agent_answer: str, correct_answer: str) -> bool:
    """
    Check if the agent's answer is correct by extracting numerical values.
    
    Args:
        agent_answer: The agent's answer (string that should end with "Answer: {value}")
        correct_answer: The correct answer (string that can be converted to float)
        
    Returns:
        True if the extracted agent's answer matches the correct answer, False otherwise
    """
    # Extract numerical value from agent_answer using regex
    pattern = r"Answer:\s*(-?[\d,]*\.?\d+)"
    match = re.search(pattern, agent_answer)
    
    if not match:
        return False 
    
    try:
        agent_numerical = _clean_number_string(match.group(1))
        correct_numerical = _clean_number_string(str(correct_answer))
        
        # Compare with small tolerance for floating point precision issues
        return abs(agent_numerical - correct_numerical) < 1e-6
    
    except (ValueError, TypeError) as e:
        print(f"Error extracting numerical values: {e}")
        return False