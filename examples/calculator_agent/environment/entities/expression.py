from typing import List, Literal, Union
from pydantic import BaseModel

class Expression(BaseModel):
    """Represents a nested arithmetic expression."""
    operation: Literal["add", "subtract", "multiply", "divide"]
    operands: List[Union['Expression', float, int]]

# Ensure forward references ('Expression') are resolved
