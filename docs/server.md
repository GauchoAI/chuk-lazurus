# Inference Server

Lazarus ships a built-in OpenAI-compatible HTTP inference server. Any tool that speaks the OpenAI API — mcp-cli, LangChain, the `openai` Python SDK, curl — works against it without modification.

## Installation

The server requires extra dependencies not included in the base install:

```bash
uv add "chuk-lazarus[server]"
# or
pip install "chuk-lazarus[server]"
```

This adds `fastapi`, `uvicorn`, and `httpx`.

## Quick Start

```bash
# Start the server with any supported model
lazarus serve --model google/gemma-3-4b-it

# Or use the dedicated standalone script
lazarus-serve --model google/gemma-3-4b-it
```

The server loads the model once, then serves all requests from it. On first run, the model is downloaded from HuggingFace Hub and cached locally.

```
Loading model: google/gemma-3-4b-it
============================================================
...
============================================================
Lazarus inference server ready
  Model     : google/gemma-3-4b-it
  Protocols : openai
  Base URL  : http://0.0.0.0:8080
  OpenAI URL: http://0.0.0.0:8080/v1
============================================================
```

## CLI Options

```bash
lazarus serve [OPTIONS]
lazarus-serve [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` / `-m` | required | HuggingFace model ID or local path |
| `--host` | `0.0.0.0` | Bind address |
| `--port` / `-p` | `8080` | Port |
| `--protocols` | `openai` | Comma-separated: `openai`, `ollama`, `anthropic` |
| `--api-key` | None | Bearer token — if set all requests must include `Authorization: Bearer <key>` |
| `--max-tokens` | `512` | Default `max_tokens` when callers do not specify one |

```bash
# With authentication
lazarus-serve --model google/gemma-3-1b-it --api-key mysecret

# Different port, multiple protocols (once implemented)
lazarus-serve --model google/gemma-3-4b-it --port 9000 --protocols openai,ollama

# Smaller model, higher token limit
lazarus-serve --model google/gemma-3-1b-it --max-tokens 2048
```

## Endpoints

### Health

```
GET /health
```

```json
{
  "status": "ok",
  "model": "google/gemma-3-4b-it",
  "protocols": ["openai"]
}
```

### OpenAI — List Models

```
GET /v1/models
```

```json
{
  "object": "list",
  "data": [
    { "id": "google/gemma-3-4b-it", "object": "model", "owned_by": "lazarus" }
  ]
}
```

### OpenAI — Chat Completions

```
POST /v1/chat/completions
```

Full OpenAI `ChatCompletion` schema, including:

- `messages` — system, user, assistant, and tool roles
- `stream` — `true` for Server-Sent Events streaming
- `tools` — function definitions for tool calling
- `max_tokens`, `temperature`, `top_p`, `stop`

#### Non-streaming

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-4b-it",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "created": 1750000000,
  "model": "google/gemma-3-4b-it",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "The capital of France is Paris."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 14, "completion_tokens": 9, "total_tokens": 23}
}
```

#### Streaming

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "google/gemma-3-4b-it", "messages": [...], "stream": true}'
```

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":""},...}]}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"The capital"},...}]}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":" of France"},...}]}
...
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

## Tool Calling

The server supports OpenAI-style function calling. Tool definitions are injected into the model's chat template via `tokenizer.apply_chat_template(..., tools=[...])`. The model's `<tool_call>` output blocks are parsed and returned in OpenAI format.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-4b-it",
    "messages": [{"role": "user", "content": "What time is it?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_current_time",
        "description": "Get the current time",
        "parameters": {
          "type": "object",
          "properties": {},
          "required": []
        }
      }
    }]
  }'
```

When the model calls a tool the response has `finish_reason: "tool_calls"`:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_a1b2c3",
        "type": "function",
        "function": {"name": "get_current_time", "arguments": "{}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Send the tool result back as a `tool` role message:

```json
{
  "messages": [
    {"role": "user", "content": "What time is it?"},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_a1b2c3", ...}]},
    {"role": "tool", "tool_call_id": "call_a1b2c3", "content": "14:32 UTC"}
  ]
}
```

## mcp-cli Integration

[mcp-cli](https://github.com/chrishayuk/mcp-cli) connects to the Lazarus server as an OpenAI-compatible provider:

```bash
# Start the server
lazarus-serve --model google/gemma-3-4b-it --api-key lazarus

# In another terminal, start mcp-cli
mcp-cli chat \
  --provider lazarus \
  --server time \
  --model google/gemma-3-4b-it

# With dashboard
mcp-cli chat \
  --provider lazarus \
  --server time \
  --model google/gemma-3-4b-it \
  --dashboard
```

mcp-cli discovers the `lazarus` provider from its config. The provider entry points to `http://localhost:8080/v1` with the model name matching what the server is serving.

## Using the openai Python SDK

Because the server is fully OpenAI-compatible, the `openai` package works directly:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="lazarus",  # any non-empty string if auth is disabled
)

# Non-streaming
response = client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Write a haiku about Python."}],
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Count to ten."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Protocol Architecture

The server uses a layered protocol router design:

```
HTTP request
    │
    ▼
FastAPI app (app.py)
    │
    ├── /v1/*  →  OpenAI router   (routers/openai.py)
    ├── /api/* →  Ollama router   (routers/ollama.py)   ← TODO
    └── /v1/*  →  Anthropic router (routers/anthropic.py) ← TODO
         │
         ▼
    ModelEngine (engine.py)          ← format-agnostic
         │
         ├── agenerate()   → InternalResponse
         └── astream()     → AsyncIterator[InternalChunk]
              │
              ▼
    UnifiedPipeline (inference/)
```

Each router translates its wire format to `InternalRequest`, calls the engine, and translates `InternalResponse` back. The engine knows nothing about protocols.

### Planned Protocols

| Protocol | Status | Endpoints |
|----------|--------|-----------|
| OpenAI | Implemented | `GET /v1/models`, `POST /v1/chat/completions` |
| Ollama | Planned | `GET /api/tags`, `POST /api/chat`, `POST /api/generate` |
| Anthropic | Planned | `POST /v1/messages` |

Enable multiple protocols when they are implemented:

```bash
lazarus-serve --model gemma-3-1b-it --protocols openai,ollama,anthropic
```

## Programmatic Usage

You can embed the server in your own application:

```python
import asyncio
from chuk_lazarus.server import ModelEngine, Protocol, create_app
import uvicorn

async def main():
    engine = await ModelEngine.load("google/gemma-3-1b-it")
    app = create_app(
        engine,
        protocols=[Protocol.OPENAI],
        api_key="secret",
        default_max_tokens=1024,
    )
    config = uvicorn.Config(app, host="0.0.0.0", port=8080)
    server = uvicorn.Server(config)
    await server.serve()

asyncio.run(main())
```

## See Also

- [Client Library](client.md) — Python client for the server
- [Inference Guide](inference.md) — Direct pipeline usage without HTTP
- [CLI Reference](cli.md) — `lazarus serve` command details
