from agentframe import function_tool


@function_tool
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello {name}!"


@function_tool(name="greet_custom", description="Custom greeting")
def hello_custom(n: str) -> str:
    return f"Hi {n}"


def describe() -> str:
    return "no params"


def _mix_func(a: int, b: str, c: float = 1.0) -> str:
    return f"{a} {b} {c}"


mix_tool = function_tool(_mix_func)


class TestFunctionToolSchema:

    def test_basic_schema(self):
        assert hello.name == "hello"
        assert hello.description == "Say hello."
        schema = hello.openai_tool
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "hello"
        assert "name" in fn["parameters"]["properties"]
        assert fn["parameters"]["required"] == ["name"]

    def test_type_mapping(self):
        fn = mix_tool.openai_tool["function"]
        props = fn["parameters"]["properties"]
        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "string"
        assert props["c"]["type"] == "number"
        assert fn["parameters"]["required"] == ["a", "b"]

    def test_custom_name_and_description(self):
        assert hello_custom.name == "greet_custom"
        assert hello_custom.description == "Custom greeting"

    def test_no_params(self):
        tool = function_tool(describe)
        assert tool.openai_tool["function"]["parameters"]["properties"] == {}
        assert tool.openai_tool["function"]["parameters"]["required"] == []


class TestFunctionToolCall:

    def test_call_success(self):
        result = hello.call(name="World")
        assert result == "Hello World!"

    def test_call_with_int(self):
        result = mix_tool.call(a=1, b="hello", c=2.5)
        assert result == "1 hello 2.5"

    def test_call_error(self):
        result = hello.call(wrong_arg="x")
        assert result.startswith("Error:")

    def test_call_missing_arg(self):
        result = hello.call()
        assert result.startswith("Error:")
