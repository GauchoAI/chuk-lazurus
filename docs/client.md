# Client Library

Lazarus ships a Python client library for the inference server. It mirrors the shape of the `openai` SDK closely enough that code using it looks familiar, but has no dependency on the OpenAI package — it uses `httpx` directly.

Both sync and async clients are included.

## Installation

The client shares the same optional dependency group as the server:

```bash
uv add "chuk-lazarus[server]"
# or
pip install "chuk-lazarus[server]"
```

This adds `httpx`.

## Quick Start

```python
from chuk_lazarus.client import LazarusClient, ChatMessage, ClientRole

client = LazarusClient(base_url="http://localhost:8080")

response = client.chat(
    model="google/gemma-3-4b-it",
    messages=[ChatMessage(role=ClientRole.USER, content="What is the capital of France?")],
)
print(response.content)  # "The capital of France is Paris."
print(f"{response.prompt_tokens} prompt / {response.completion_tokens} completion tokens")
```

## Public Types

### `ChatMessage`

A single turn in the conversation.

```python
class ChatMessage(BaseModel):
    role: ClientRole   # SYSTEM | USER | ASSISTANT
    content: str
```

### `ClientRole`

```python
class ClientRole(str, Enum):
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"
```

### `ChatResponse`

Returned by non-streaming `chat()`.

```python
class ChatResponse(BaseModel):
    id: str
    model: str
    content: str
    finish_reason: ClientFinishReason   # STOP | LENGTH
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

### `ModelInfo` / `ModelList`

Returned by `list_models()`.

```python
class ModelInfo(BaseModel):
    id: str
    owned_by: str   # "lazarus"

class ModelList(BaseModel):
    data: list[ModelInfo]
```

### `HealthResponse`

Returned by `health()`.

```python
class HealthResponse(BaseModel):
    status: str          # "ok"
    model: str           # e.g. "google/gemma-3-4b-it"
    protocols: list[str] # e.g. ["openai"]
```

## Sync Client — `LazarusClient`

```python
LazarusClient(
    base_url: str = "http://localhost:8080",
    api_key: str | None = None,
    timeout: float = 120.0,
)
```

Supports use as a context manager:

```python
with LazarusClient(base_url="http://localhost:8080", api_key="secret") as client:
    response = client.chat(model=..., messages=[...])
```

Or managed manually:

```python
client = LazarusClient()
try:
    response = client.chat(...)
finally:
    client.close()
```

### `chat()`

```python
response = client.chat(
    model="google/gemma-3-4b-it",
    messages=[
        ChatMessage(role=ClientRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=ClientRole.USER, content="Explain recursion."),
    ],
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    stop=None,
)
print(response.content)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model ID — must match what the server is serving |
| `messages` | `list[ChatMessage]` | required | Conversation history |
| `max_tokens` | `int \| None` | server default | Max tokens to generate |
| `temperature` | `float \| None` | server default | Sampling temperature |
| `top_p` | `float \| None` | server default | Nucleus sampling threshold |
| `stop` | `str \| list[str] \| None` | None | Stop sequences |

Returns `ChatResponse`.

### `stream_chat()`

```python
for chunk in client.stream_chat(
    model="google/gemma-3-4b-it",
    messages=[ChatMessage(role=ClientRole.USER, content="Count to ten.")],
):
    print(chunk, end="", flush=True)
print()
```

Yields `str` text deltas as the model generates them. Accepts the same keyword parameters as `chat()`.

### `list_models()`

```python
models = client.list_models()
for m in models.data:
    print(m.id)  # "google/gemma-3-4b-it"
```

### `health()`

```python
status = client.health()
print(status.status)    # "ok"
print(status.model)     # "google/gemma-3-4b-it"
print(status.protocols) # ["openai"]
```

## Async Client — `AsyncLazarusClient`

Drop-in async equivalent. Use `await` for `chat()`, `list_models()`, and `health()`; use `async for` for `stream_chat()`.

```python
AsyncLazarusClient(
    base_url: str = "http://localhost:8080",
    api_key: str | None = None,
    timeout: float = 120.0,
)
```

Supports async context manager:

```python
async with AsyncLazarusClient(base_url="http://localhost:8080") as client:
    response = await client.chat(
        model="google/gemma-3-4b-it",
        messages=[ChatMessage(role=ClientRole.USER, content="Hello!")],
    )
    print(response.content)
```

### Async streaming

```python
async with AsyncLazarusClient() as client:
    async for chunk in client.stream_chat(
        model="google/gemma-3-4b-it",
        messages=[ChatMessage(role=ClientRole.USER, content="Write a haiku.")],
    ):
        print(chunk, end="", flush=True)
    print()
```

### Manual lifecycle

```python
client = AsyncLazarusClient()
try:
    response = await client.chat(...)
finally:
    await client.aclose()
```

## Multi-Turn Conversations

Build conversation history by collecting assistant responses:

```python
from chuk_lazarus.client import LazarusClient, ChatMessage, ClientRole

client = LazarusClient()
model = "google/gemma-3-4b-it"

history: list[ChatMessage] = [
    ChatMessage(role=ClientRole.SYSTEM, content="You are a helpful assistant."),
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit"):
        break

    history.append(ChatMessage(role=ClientRole.USER, content=user_input))

    response = client.chat(model=model, messages=history)
    print(f"Assistant: {response.content}")

    history.append(ChatMessage(role=ClientRole.ASSISTANT, content=response.content))
```

## Error Handling

Both clients raise `httpx.HTTPStatusError` on non-2xx responses:

```python
import httpx
from chuk_lazarus.client import LazarusClient, ChatMessage, ClientRole

with LazarusClient() as client:
    try:
        response = client.chat(
            model="google/gemma-3-4b-it",
            messages=[ChatMessage(role=ClientRole.USER, content="Hello")],
        )
    except httpx.HTTPStatusError as e:
        print(f"Server error {e.response.status_code}: {e.response.text}")
    except httpx.TimeoutException:
        print("Request timed out")
```

## See Also

- [Inference Server](server.md) — Starting and configuring the server
- [Inference Guide](inference.md) — Direct pipeline usage without HTTP
- [CLI Reference](cli.md) — `lazarus serve` command details
