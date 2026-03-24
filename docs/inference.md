# Inference Guide

Run text generation with pretrained models from HuggingFace Hub using the unified inference pipeline.

## Quick Start

### UnifiedPipeline (Recommended)

The `UnifiedPipeline` auto-detects model family and provides a simplified, one-liner API:

```python
from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig, DType

# One-liner model loading - auto-detects family!
pipeline = UnifiedPipeline.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Simple chat API
result = pipeline.chat("What is the capital of France?")
print(result.text)
print(result.stats.summary)  # "25 tokens in 0.42s (59.5 tok/s)"
```

### With Custom Configuration

```python
from chuk_lazarus.inference import (
    UnifiedPipeline,
    UnifiedPipelineConfig,
    GenerationConfig,
    DType,
)

# Configure the pipeline
config = UnifiedPipelineConfig(
    dtype=DType.BFLOAT16,
    default_system_message="You are a helpful coding assistant.",
    default_max_tokens=200,
    default_temperature=0.7,
)

pipeline = UnifiedPipeline.from_pretrained(
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    pipeline_config=config,
)

# Generate with custom settings
result = pipeline.chat(
    "Write a Python function to calculate Fibonacci numbers",
    max_new_tokens=300,
    temperature=0.3,
)
print(result.text)
```

### CLI Inference

```bash
# Basic inference with TinyLlama
chuk-lazarus infer --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --prompt "What is the capital of France?"

# With generation parameters
chuk-lazarus infer \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --prompt "Explain quantum computing in one sentence" \
  --max-tokens 100 \
  --temperature 0.7
```

### Low-Level Python API

For more control, use the models directly:

```python
from chuk_lazarus.models_v2 import LlamaConfig, LlamaForCausalLM
from transformers import AutoTokenizer
import mlx.core as mx

# Load model and tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = LlamaConfig.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, config)

# Generate text
prompt = "What is machine learning?"
input_ids = tokenizer.encode(prompt, return_tensors="np")
input_ids = mx.array(input_ids)

output_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    stop_tokens=[tokenizer.eos_token_id],
)
mx.eval(output_ids)

response = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
print(response)
```

## UnifiedPipeline API

### Core Classes

| Class | Description |
|-------|-------------|
| `UnifiedPipeline` | High-level API with auto-detection and generation |
| `UnifiedPipelineConfig` | Pipeline configuration (dtype, defaults, introspection) |
| `GenerationConfig` | Generation parameters (max_tokens, temperature, top_p) |
| `GenerationResult` | Generation output with text and stats |
| `ChatHistory` | Multi-turn conversation management |

### Loading Models

```python
# Synchronous loading (auto-detects family)
pipeline = UnifiedPipeline.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    pipeline_config=UnifiedPipelineConfig(dtype=DType.BFLOAT16),
)

# Async loading
pipeline = await UnifiedPipeline.from_pretrained_async(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

# Access detected family
print(f"Family: {pipeline.family_type}")  # ModelFamilyType.LLAMA
```

### Supported Model Families

The pipeline auto-detects these model families:

| Family | Model Types | Example Models |
|--------|-------------|----------------|
| `llama` | llama, mistral, codellama | TinyLlama, Llama 2/3, SmolLM2 |
| `llama4` | llama4 | Llama 4 Scout |
| `gemma` | gemma, gemma2, gemma3 | Gemma 3 270M-27B |
| `granite` | granite | Granite 3.x |
| `granitemoehybrid` | granitemoehybrid | Granite 4.0 |
| `jamba` | jamba | Jamba, Jamba 1.5 |
| `mamba` | mamba | Mamba SSM models |
| `starcoder2` | starcoder2 | StarCoder2 3B/7B/15B |
| `qwen3` | qwen2, qwen3 | Qwen 2/3 |
| `gpt2` | gpt2 | GPT-2, DistilGPT-2 |

### Chat API

```python
# Simple single-turn chat
result = pipeline.chat("What is 2+2?")

# With custom system message
result = pipeline.chat(
    "Write a haiku",
    system_message="You are a poet.",
)

# Multi-turn conversation
from chuk_lazarus.inference import ChatHistory

history = ChatHistory()
history.add_system("You are a helpful assistant.")
history.add_user("What is Python?")
history.add_assistant("Python is a programming language.")
history.add_user("What is it used for?")

result = pipeline.chat_with_history(history)
```

### Raw Generation

```python
# Direct prompt without chat formatting
result = pipeline.generate(
    "Once upon a time",
    max_new_tokens=100,
    temperature=0.9,
)

# With full config
from chuk_lazarus.inference import GenerationConfig

config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
)
result = pipeline.generate("The quick brown fox", config=config)
```

### Streaming Generation

```python
from chuk_lazarus.inference import generate_stream

# Stream tokens as they're generated
for chunk in generate_stream(model, tokenizer, "Write a story"):
    print(chunk, end="", flush=True)
```

## Stateful Context Engines

### EngineMode and make_engine()

Select the `KV_DIRECT` engine at pipeline construction time and retrieve a ready-to-use generator:

```python
from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig
from chuk_lazarus.inference.unified import EngineMode

# Load with KV-direct engine mode selected
config = UnifiedPipelineConfig(engine=EngineMode.KV_DIRECT)
pipeline = UnifiedPipeline.from_pretrained("google/gemma-3-1b-it", config=config)

# Get a ready-to-use KVDirectGenerator for any supported model
kv_gen = pipeline.make_engine()
```

### KVDirectGenerator

Stores K,V directly after prefill (post-norm, post-RoPE). Bit-exact with standard KV cache. Exposes the full generation lifecycle — prefill, step, extend, and slide — for direct KV store control:

```python
import mlx.core as mx
from chuk_lazarus.inference.context import make_kv_generator

# Auto-detects model family (Gemma, Llama, Mistral, ...)
kv_gen = make_kv_generator(pipeline.model)

# Prefill on a prompt
prompt_ids = mx.array([[tok1, tok2, ...]])  # (1, S)
logits, kv_store = kv_gen.prefill(prompt_ids)

# Generate tokens one at a time
seq_len = prompt_ids.shape[1]
for _ in range(max_new_tokens):
    next_token = mx.argmax(logits[0, -1])
    logits, kv_store = kv_gen.step(next_token[None, None], kv_store, seq_len)
    seq_len += 1

# Extend with a new batch of tokens (e.g. second turn)
new_ids = mx.array([[...]])  # (1, N)
logits, kv_store = kv_gen.extend(new_ids, kv_store, abs_start=seq_len)

# Evict oldest tokens when budget is hit
kv_store = kv_gen.slide(kv_store, evict_count=64)

# Memory accounting
print(f"KV bytes at 512 tokens: {kv_gen.kv_bytes(512):,}")
```

### Custom model support

Use the auto-detect factory or construct from a specific adapter. To support a new architecture, implement `ModelBackboneProtocol` and pass it to `KVDirectGenerator`:

```python
from chuk_lazarus.inference.context import (
    KVDirectGenerator,
    make_kv_generator,
    GemmaBackboneAdapter,
    LlamaBackboneAdapter,
    ModelBackboneProtocol,
    TransformerLayerProtocol,
)

# Auto-detect factory (Gemma, Llama, Mistral):
gen = make_kv_generator(model)

# Or construct from a specific adapter:
gen = KVDirectGenerator.from_gemma_rs(rs_model)
gen = KVDirectGenerator.from_llama(llama_model)

# For a new architecture, implement ModelBackboneProtocol and pass it in:
class MyBackboneAdapter:  # implements ModelBackboneProtocol
    ...
gen = KVDirectGenerator(MyBackboneAdapter(my_model))
```

### CLI

The `--engine` flag selects the engine at the CLI level:

```bash
# Standard generation (default)
lazarus infer --model google/gemma-3-1b-it --prompt "Hello"

# KV-direct stateful engine
lazarus infer --model google/gemma-3-1b-it --prompt "Hello" --engine kv_direct
```

### Choosing an engine

| Engine | Class | Use case |
|--------|-------|----------|
| `standard` | Built-in KV cache | General inference, default |
| `kv_direct` | `KVDirectGenerator` | Stateful multi-turn, sliding window, custom eviction |
| `bounded_kv` | `BoundedKVEngine` | HOT/WARM/COLD memory budgets (Gemma) |
| `unlimited` | `UnlimitedContextEngine` | Unbounded context via checkpoint replay |

## Example Scripts

The `examples/inference/` directory contains streamlined examples using UnifiedPipeline:

```bash
# Simple inference (any supported model)
uv run python examples/inference/simple_inference.py --prompt "What is the capital of France?"

# List supported families
uv run python examples/inference/simple_inference.py --list-families

# Llama family with model presets
uv run python examples/inference/llama_inference.py --model smollm2-360m
uv run python examples/inference/llama_inference.py --list

# Gemma 3 with interactive chat
uv run python examples/inference/gemma_inference.py --chat

# Granite (IBM)
uv run python examples/inference/granite_inference.py --model granite-3.1-2b

# Llama 4 Scout (Mamba-Transformer hybrid)
uv run python examples/inference/llama4_inference.py

# StarCoder2 (code generation)
uv run python examples/inference/starcoder2_inference.py --prompt "def quicksort(arr):"

# Jamba (hybrid Mamba-Transformer MoE)
uv run python examples/inference/jamba_inference.py --test-tiny
```

## Llama Family Inference

The `examples/inference/llama_inference.py` script provides a unified interface for Llama-architecture models:

```bash
# List available model presets
uv run python examples/inference/llama_inference.py --list

# Run with different models
uv run python examples/inference/llama_inference.py --model tinyllama
uv run python examples/inference/llama_inference.py --model smollm2-360m
uv run python examples/inference/llama_inference.py --model llama3.2-1b

# Custom prompt
uv run python examples/inference/llama_inference.py \
  --model smollm2-360m \
  --prompt "Explain relativity in simple terms" \
  --max-tokens 150 \
  --temperature 0.8
```

### Available Model Presets

| Preset | Model ID | Parameters | Notes |
|--------|----------|------------|-------|
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | Fast, good for testing |
| `smollm2-135m` | HuggingFaceTB/SmolLM2-135M-Instruct | 135M | Tiny, runs anywhere |
| `smollm2-360m` | HuggingFaceTB/SmolLM2-360M-Instruct | 360M | Good quality/speed balance |
| `smollm2-1.7b` | HuggingFaceTB/SmolLM2-1.7B-Instruct | 1.7B | High quality, still fast |
| `llama2-7b` | meta-llama/Llama-2-7b-chat-hf | 7B | Llama 2 Chat |
| `llama2-13b` | meta-llama/Llama-2-13b-chat-hf | 13B | Larger Llama 2 |
| `llama3.2-1b` | meta-llama/Llama-3.2-1B-Instruct | 1B | Smallest Llama 3 |
| `llama3.2-3b` | meta-llama/Llama-3.2-3B-Instruct | 3B | Small but capable |
| `llama3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | 8B | Standard size |
| `mistral-7b` | mistralai/Mistral-7B-Instruct-v0.3 | 7B | Sliding window attention |

**Note:** Meta Llama models require HuggingFace authentication. Run `huggingface-cli login` first.

## Generation Parameters

The `generate()` method supports several parameters to control text generation:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_new_tokens` | int | 100 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0 = greedy, higher = more random) |
| `top_p` | float | 0.9 | Nucleus sampling probability threshold |
| `top_k` | int | None | Top-k sampling (limits to k most likely tokens) |
| `repetition_penalty` | float | 1.0 | Penalty for repeating tokens (>1 reduces repetition) |
| `stop_tokens` | list | None | Token IDs that stop generation |

### Example: Temperature Effects

```python
# Greedy decoding (deterministic)
output = model.generate(input_ids, temperature=0.0)

# Low temperature (focused, coherent)
output = model.generate(input_ids, temperature=0.3)

# Medium temperature (balanced)
output = model.generate(input_ids, temperature=0.7)

# High temperature (creative, diverse)
output = model.generate(input_ids, temperature=1.2)
```

## Weight Loading

Models are loaded from HuggingFace Hub and automatically converted:

1. **Download**: Weights are downloaded via `huggingface_hub.snapshot_download()`
2. **Load**: MLX's native `mx.load()` handles safetensors files
3. **Detect**: Model family is auto-detected from config.json
4. **Convert**: Weights are mapped using family-specific converters
5. **Update**: Weights are loaded into the model via `model.update()`

### Dtype Considerations

Use `bfloat16` for numerical stability with most models:

```python
config = UnifiedPipelineConfig(dtype=DType.BFLOAT16)  # Recommended
```

## Chat Templates

The pipeline uses the tokenizer's built-in chat template when available:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

# Apply chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

## Performance Tips

1. **Use bfloat16**: Default dtype for numerical stability
2. **Enable KV-cache**: Automatically enabled for autoregressive generation
3. **Batch prompts**: Process multiple prompts in a single forward pass when possible
4. **Use smaller models**: SmolLM2-135M for fast iteration, larger models for quality
5. **Use KVDirectGenerator for stateful inference**: `make_kv_generator(model)` gives you direct control over the KV store — sliding window eviction, turn-level extend, and memory accounting without touching the model code.

## Troubleshooting

### Garbage Output

If the model produces nonsensical output:
- Ensure weights loaded correctly (check tensor count)
- Verify dtype is `bfloat16` (not `float16`)
- Check that `tie_word_embeddings` matches config

### Slow Generation

- First token is slow (model compilation)
- Subsequent tokens should be fast (~50-150 tok/s)
- Larger models need more memory bandwidth

### Missing Weights

If weight loading fails:
- Check model files exist in cache
- Verify safetensors format
- Some models may need HF authentication

### Unsupported Model Family

If you get "Unable to detect model family":
- Check if the model's `model_type` is supported
- The model architecture may not be implemented yet
- Open an issue to request support for new model families

## Gemma Inference

Gemma 3 is Google's latest open model family with 5 sizes (270M, 1B, 4B, 12B, 27B) and 128K context.

```bash
# Basic inference
uv run python examples/inference/gemma_inference.py --prompt "What is the capital of France?"

# Gemma 3 270M (smallest, fastest)
uv run python examples/inference/gemma_inference.py --model gemma3-270m

# Interactive chat mode
uv run python examples/inference/gemma_inference.py --chat
```

## Granite Inference

IBM Granite models are available in dense (3.x) and hybrid (4.x) variants:

```bash
# Granite 3.1 (dense)
uv run python examples/inference/granite_inference.py --model granite-3.1-2b

# Granite 4.0 (hybrid Mamba/Transformer)
uv run python examples/inference/granite_inference.py --model granite-4.0-micro

# Test without downloading
uv run python examples/inference/granite_inference.py --test-tiny
```

## Jamba Inference

Jamba is AI21 Labs' hybrid Mamba-Transformer MoE model with 256K context:

```bash
# Test with tiny model
uv run python examples/inference/jamba_inference.py --test-tiny

# Basic inference
uv run python examples/inference/jamba_inference.py --prompt "What is quantum computing?"
```

## StarCoder2 Inference

BigCode's code generation models:

```bash
# Code completion
uv run python examples/inference/starcoder2_inference.py --prompt "def fibonacci(n):"

# Use specific model size
uv run python examples/inference/starcoder2_inference.py --model starcoder2-7b
```

## Serving Models

To expose a model over HTTP as an OpenAI-compatible API — for use with mcp-cli, LangChain, the `openai` SDK, or any other OpenAI-compatible client — start the built-in inference server:

```bash
# Requires the server extra
uv add "chuk-lazarus[server]"

lazarus serve --model google/gemma-3-4b-it
```

The server is ready at `http://localhost:8080/v1`. From there:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "google/gemma-3-4b-it", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Or use the Python client library:

```python
from chuk_lazarus.client import LazarusClient, ChatMessage, ClientRole

with LazarusClient() as client:
    response = client.chat(
        model="google/gemma-3-4b-it",
        messages=[ChatMessage(role=ClientRole.USER, content="Hello!")],
    )
    print(response.content)
```

See [server.md](server.md) for the full server guide and [client.md](client.md) for the client library.

## See Also

- [Models Guide](models.md) - Architecture details
- [Training Guide](training.md) - Fine-tuning models
- [Inference Server](server.md) - OpenAI-compatible HTTP server
- [Client Library](client.md) - Python client for the server
- [Examples](../examples/inference/) - Working inference examples
