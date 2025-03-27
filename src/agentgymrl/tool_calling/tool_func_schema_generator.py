import inspect
import warnings 
from typing import Any, Callable, Dict

from pydantic import BaseModel, Field, create_model
from pydantic_core import PydanticUndefined



def generate_function_schema(func: Callable[..., Any]) -> Dict[str, Any]:
    """
    Generates a JSON schema dictionary representing a function call based on its signature.

    Relies on type hints for parameters. Parameters without type hints
    (excluding 'self' or 'cls' for methods) will cause a TypeError.
    Handles standard Python types, Pydantic models, Literals, Unions,
    Optionals, Lists, Dicts, etc., supported by Pydantic.

    Args:
        func: The function, method, or callable to generate a schema for.

    Returns:
        A dictionary representing the JSON schema for the function call,
        following a structure similar to OpenAI Functions or Tool Calls.

    Raises:
        TypeError: If any parameter lacks a type hint (excluding self/cls).
        RuntimeError: If Pydantic model creation or schema generation fails.

    Warnings:
        SchemaGenerationWarning: If VAR_POSITIONAL (*args) or VAR_KEYWORD (**kwargs)
                                 parameters are encountered, as they cannot be
                                 directly represented in a fixed JSON schema.
    """
    try:
        sig = inspect.signature(func)
        func_name = func.__name__
        func_desc = inspect.getdoc(func) or f"Schema for function '{func_name}'."
    except ValueError as exc:
        raise ValueError(f"Could not determine signature for function '{func}'.") from exc
    except TypeError as exc:
        raise TypeError(f"Input must be a callable function or method, got {type(func)}.") from exc


    field_definitions: Dict[str, Any] = {}
    skipped_params = []

    is_first_param = True
    for name, param in sig.parameters.items():
        if is_first_param and name in ('self', 'cls'):
            skipped_params.append(name)
            is_first_param = False
            continue

        is_arg_or_kwarg = param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        if is_arg_or_kwarg:
            warnings.warn(
                f"Skipping {param.kind} parameter '{name}' in '{func_name}' as it doesn't map to a fixed JSON schema property.",
            )
            skipped_params.append(name)
            continue

        type_hint_not_found = param.annotation == inspect.Parameter.empty
        if type_hint_not_found:
            raise TypeError(
                f"Parameter '{name}' in function '{func_name}' lacks a type hint. "
                f"Schema generation requires type hints for all mapped parameters."
            )

        if param.default == inspect.Parameter.empty:
            default_value = PydanticUndefined
        else:
            default_value = param.default

        field_definitions[name] = (
            param.annotation,
            Field(default=default_value, description=f"Parameter '{name}' for function '{func_name}'")
        )

    only_skipped_params_or_no_args = len(field_definitions) == 0
    if only_skipped_params_or_no_args:
        # According to OpenAI spec, parameters should be an object, even if empty
        params_schema = {"type": "object", "properties": {}}
    else:
        try:
            DynamicParamsModel = create_model(
                f'{func_name}__Params',
                **field_definitions,
                __base__=BaseModel
            )
            # Rebuild model to resolve any forward references within annotations
            # and generate the schema correctly
            DynamicParamsModel.model_rebuild(force=True)

            # Generate the JSON schema for the parameters model
            params_schema = DynamicParamsModel.model_json_schema(by_alias=False)

        except Exception as e:
            error_message = f"Could not generate schema for function '{func_name}'. Pydantic model creation or schema generation failed: {e}"
            raise RuntimeError(error_message) from e

    output_schema = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": func_desc,
            "parameters": params_schema
        }
    }

    return output_schema
