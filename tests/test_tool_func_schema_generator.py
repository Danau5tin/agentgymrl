import unittest
from typing import Dict, List, Union, Literal, Optional
from pydantic import BaseModel

from agentgymrl.tool_calling.tool_func_schema_generator import generate_function_schema


class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class UserProfile(BaseModel):
    user_id: int
    name: str
    email: Optional[str] = None
    address: Optional[Address] = None

class Expression(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"]
    operands: List[Union['Expression', float, int]]
Expression.model_rebuild()


def calculator(operation: Literal["add", "subtract", "multiply", "divide"],
               operands: List[Union[Expression, float, int]]) -> float:
    """Performs math computations."""
    pass

def get_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Gets the current weather for a location."""
    return f"Weather in {location} is nice, {unit}."

def update_user_profile(user_id: int, profile_data: UserProfile) -> bool:
    """Updates a user's profile information."""
    return True

def get_server_status() -> Dict[str, str]:
    """Returns the current server status."""
    return {"status": "ok"}

class Greeter:
    """Simple class for greeting."""
    def __init__(self, greeting: str = "Hello"):
        self.greeting = greeting

    def greet(self, name: str, enthusiasm: int = 1) -> str:
        """Greets someone with optional enthusiasm."""
        punctuation = "!" * enthusiasm
        return f"{self.greeting}, {name}{punctuation}"

    @classmethod
    def create_standard_greeter(cls) -> 'Greeter':
        """Creates a standard greeter instance."""
        return cls()

    def process_items(self, *items: str, **options: bool) -> None:
        """Processes items with options."""
        pass


class TestFunctionSchemaGenerator(unittest.TestCase):

    def test_generate_schema_calculator(self):
        """Tests schema generation for the calculator function."""
        schema = generate_function_schema(calculator)

        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "calculator")
        self.assertEqual(schema["function"]["description"], "Performs math computations.")

        params = schema["function"]["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("operation", params["properties"])
        self.assertIn("operands", params["properties"])
        self.assertCountEqual(params["required"], ["operation", "operands"])

        op_prop = params["properties"]["operation"]
        self.assertEqual(op_prop["type"], "string")
        self.assertSetEqual(set(op_prop.get("enum", [])), {"add", "subtract", "multiply", "divide"})
        self.assertIn("Parameter 'operation'", op_prop.get("description", ""))

        opd_prop = params["properties"]["operands"]
        self.assertEqual(opd_prop["type"], "array")
        self.assertIn("items", opd_prop)
        self.assertIn("anyOf", opd_prop["items"])
        self.assertEqual(len(opd_prop["items"]["anyOf"]), 3)

        if "$defs" not in params:
            self.fail("Missing $defs in parameters")

        self.assertTrue(any("$ref" in item and "Expression" in item["$ref"] for item in opd_prop["items"]["anyOf"]))
        self.assertTrue(any("type" in item and item["type"] == "number" for item in opd_prop["items"]["anyOf"])) 
        self.assertTrue(any("type" in item and item["type"] == "integer" for item in opd_prop["items"]["anyOf"]))
        self.assertIn("Parameter 'operands'", opd_prop.get("description", ""))

        self.assertIn("$defs", params)
        self.assertIn("Expression", params["$defs"])
        self.assertEqual(params["$defs"]["Expression"]["type"], "object")
        self.assertIn("operation", params["$defs"]["Expression"]["properties"])
        self.assertIn("operands", params["$defs"]["Expression"]["properties"])


    def test_generate_schema_get_weather(self):
        """Tests schema generation for a function with defaults and Literals."""
        schema = generate_function_schema(get_weather)

        self.assertEqual(schema["function"]["name"], "get_weather")
        self.assertEqual(schema["function"]["description"], "Gets the current weather for a location.")

        params = schema["function"]["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("location", params["properties"])
        self.assertIn("unit", params["properties"])
        self.assertEqual(params["required"], ["location"])

        loc_prop = params["properties"]["location"]
        self.assertEqual(loc_prop["type"], "string")
        self.assertIn("Parameter 'location'", loc_prop.get("description", ""))

        unit_prop = params["properties"]["unit"]
        self.assertEqual(unit_prop["type"], "string")
        self.assertSetEqual(set(unit_prop.get("enum", [])), {"celsius", "fahrenheit"})
        self.assertEqual(unit_prop.get("default"), "celsius")
        self.assertIn("Parameter 'unit'", unit_prop.get("description", ""))


    def test_generate_schema_update_user_profile(self):
        """Tests schema generation with nested Pydantic models."""
        schema = generate_function_schema(update_user_profile)

        self.assertEqual(schema["function"]["name"], "update_user_profile")
        self.assertEqual(schema["function"]["description"], "Updates a user's profile information.")

        params = schema["function"]["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("user_id", params["properties"])
        self.assertIn("profile_data", params["properties"])
        self.assertCountEqual(params["required"], ["user_id", "profile_data"])

        uid_prop = params["properties"]["user_id"]
        self.assertEqual(uid_prop["type"], "integer")
        self.assertIn("Parameter 'user_id'", uid_prop.get("description", ""))

        pd_prop = params["properties"]["profile_data"]

        if  "$defs" not in params: 
            self.fail("Missing $defs in parameters")

        self.assertIn("$ref", pd_prop)
        self.assertIn("UserProfile", pd_prop["$ref"])
        self.assertIn("Parameter 'profile_data'", pd_prop.get("description", ""))

        self.assertIn("$defs", params)
        self.assertIn("Address", params["$defs"])
        self.assertIn("UserProfile", params["$defs"])

        up_def = params["$defs"]["UserProfile"]
        self.assertEqual(up_def["type"], "object")
        self.assertIn("user_id", up_def["properties"])
        self.assertIn("name", up_def["properties"])
        self.assertIn("email", up_def["properties"])
        self.assertIn("address", up_def["properties"])
        self.assertCountEqual(up_def.get("required", []), ["user_id", "name"]) # email and address are Optional

        addr_ref = up_def["properties"]["address"]
        self.assertIn("anyOf", addr_ref)
        self.assertEqual(len(addr_ref["anyOf"]), 2)
        self.assertTrue(any("$ref" in item and "Address" in item["$ref"] for item in addr_ref["anyOf"]))
        self.assertTrue(any("type" in item and item["type"] == "null" for item in addr_ref["anyOf"]))


    def test_generate_schema_no_arguments(self):
        """Tests schema generation for a function with no arguments."""
        schema = generate_function_schema(get_server_status)

        self.assertEqual(schema["function"]["name"], "get_server_status")
        self.assertEqual(schema["function"]["description"], "Returns the current server status.")

        params = schema["function"]["parameters"]
        expected_params = {"type": "object", "properties": {}}
        if "$defs" in params:
            expected_params["$defs"] = params["$defs"]
        self.assertEqual(params, expected_params)


    def test_generate_schema_class_method_unbound(self):
        """Tests schema generation skipping 'self' on an unbound method."""
        schema = generate_function_schema(Greeter.greet) 

        self.assertEqual(schema["function"]["name"], "greet")
        self.assertEqual(schema["function"]["description"], "Greets someone with optional enthusiasm.")

        params = schema["function"]["parameters"]
        self.assertNotIn("self", params.get("properties", {}))
        self.assertIn("name", params["properties"])
        self.assertIn("enthusiasm", params["properties"])
        self.assertEqual(params["required"], ["name"]) # Enthusiasm has a default

        name_prop = params["properties"]["name"]
        self.assertEqual(name_prop["type"], "string")

        enth_prop = params["properties"]["enthusiasm"]
        self.assertEqual(enth_prop["type"], "integer")
        self.assertEqual(enth_prop.get("default"), 1)


    def test_generate_schema_class_method_bound(self):
        """Tests schema generation skipping 'self' on a bound method."""
        greeter_instance = Greeter()
        schema = generate_function_schema(greeter_instance.greet) 

        self.assertEqual(schema["function"]["name"], "greet")
        self.assertEqual(schema["function"]["description"], "Greets someone with optional enthusiasm.")

        params = schema["function"]["parameters"]
        self.assertNotIn("self", params.get("properties", {})) 
        self.assertIn("name", params["properties"])
        self.assertIn("enthusiasm", params["properties"])
        self.assertEqual(params["required"], ["name"])

        name_prop = params["properties"]["name"]
        self.assertEqual(name_prop["type"], "string")

        enth_prop = params["properties"]["enthusiasm"]
        self.assertEqual(enth_prop["type"], "integer")
        self.assertEqual(enth_prop.get("default"), 1)


    def test_generate_schema_classmethod(self):
        """Tests schema generation skipping 'cls' on a classmethod."""
        schema = generate_function_schema(Greeter.create_standard_greeter) 

        self.assertEqual(schema["function"]["name"], "create_standard_greeter")
        self.assertEqual(schema["function"]["description"], "Creates a standard greeter instance.")

        params = schema["function"]["parameters"]
        self.assertNotIn("cls", params.get("properties", {})) 
        expected_params = {"type": "object", "properties": {}}
        if "$defs" in params:
            expected_params["$defs"] = params["$defs"]
        self.assertEqual(params, expected_params)


    def test_missing_type_hint_raises_error(self):
        """Tests that a function with missing type hints raises TypeError."""
        def func_missing_hint(arg1, arg2: int):
            pass

        with self.assertRaises(TypeError) as cm:
            generate_function_schema(func_missing_hint)
        self.assertIn("Parameter 'arg1'", str(cm.exception))
        self.assertIn("lacks a type hint", str(cm.exception))

    def test_skip_args_kwargs_with_warning(self):
        """Tests that *args and **kwargs are skipped and a warning is issued."""
        greeter_instance = Greeter()

        # Use assertWarns context manager
        schema = generate_function_schema(greeter_instance.process_items)

        # Check that the resulting schema has no parameters
        self.assertEqual(schema["function"]["name"], "process_items")
        params = schema["function"]["parameters"]
        expected_params = {"type": "object", "properties": {}}
        # Allow for potential $defs from pydantic v2, even if empty
        if "$defs" in params:
            expected_params["$defs"] = params["$defs"]
        self.assertEqual(params, expected_params)


    def test_invalid_input_type(self):
        """Tests that non-callable input raises TypeError."""
        with self.assertRaises(TypeError) as cm:
            generate_function_schema(123)
        self.assertIn("Input must be a callable function or method", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
