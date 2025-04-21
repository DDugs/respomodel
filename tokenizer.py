def custom_tokenizer(text):
    """
    Custom tokenizer for the Symptoms column.
    Splits text on commas, handles extra whitespace, and normalizes tokens.
    
    Args:
        text (str): Input string of symptoms (e.g., "Fever, Rapid breathing")
        
    Returns:
        list: List of cleaned and normalized symptom tokens
    """
    if not isinstance(text, str):
        return []  # Handle non-string inputs
    tokens = [token.strip().lower() for token in text.split(',')]
    tokens = [token.replace(' ', '_') for token in tokens if token]
    return tokens
