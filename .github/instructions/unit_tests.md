---
name: Test Writing Standards
description: Clean test writing guidelines for ux-agent
applyTo: "tests/**/*.py"
---

# Test Writing Standards

## Type Hints

Always use complete type hints for fixtures and test functions:

```python
@pytest.fixture
def executor(registry: ToolRegistry) -> ToolExecutor:
    return ToolExecutor(registry)

@pytest.mark.asyncio
async def test_feature(self, executor: ToolExecutor) -> None:
    result = await executor.execute("action", {})
    assert result == "expected"
```

## Test Organization

Group related tests using classes with `Test` prefix:

```python
class TestPydanticConversion:
    async def test_simple_model(self): ...
    async def test_nested_model(self): ...

class TestEnumConversion:
    async def test_standalone_enum(self): ...
    async def test_invalid_value(self): ...
```

## Naming

Use descriptive test names that explain what is being tested:

```python
# Good
async def test_pydantic_with_nested_optional(self): ...
async def test_invalid_enum_value(self): ...

# Bad
async def test_case_1(self): ...
async def test_model(self): ...
```

## No Comments

Write self-documenting code without comments or docstrings. Test names and code should be clear enough.

## Test Data

Define test models at module level:

```python
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"

class Task(BaseModel):
    title: str
    priority: Priority = Priority.MEDIUM
```

## Assertions

Be specific in assertions:

```python
# Good
assert "Berlin" in result
assert len(results) == 2

# Bad
assert result
assert True
```

## Exception Testing

Use pytest.raises with match pattern:

```python
with pytest.raises(ValueError, match="not found"):
    await executor.execute("nonexistent", {})
```

## Async Tests

Always use `@pytest.mark.asyncio` decorator for async tests:

```python
@pytest.mark.asyncio
async def test_async_operation(self, executor: ToolExecutor) -> None:
    result = await executor.execute("action", {})
    assert isinstance(result, str)
```
