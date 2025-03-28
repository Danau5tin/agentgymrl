import json
import os
from typing import Literal

from agentgymrl.tool_calling.tool_func_schema_generator import generate_function_schema

from environment.tools.calculator import calculate

current_file_dir = os.path.dirname(os.path.abspath(__file__))


def get_sys_msg(variant: Literal["no_tools", "phi_4_tools"]) -> str:
    variant_path = ""
    if variant == "no_tools":
        variant_path = "./sys_msg_no_tools.md"
    elif variant == "phi_4_tools":
        variant_path = "./sys_msg_phi_4_tools.md"

    if not variant_path:
        raise ValueError(f"Unknown variant: {variant}")
    
    sys_msg_path = os.path.join(current_file_dir, variant_path)

    with open(sys_msg_path, "r", encoding="utf-8") as f:
        return f.read()
    


schema = generate_function_schema(calculate)
with open("schema.json", "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)
