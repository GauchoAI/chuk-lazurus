## Part 7: Limitations & Future Work

### Current Limitations

1. **Steering is simulated** - Actual activation patching needed to validate
2. **Small dataset** - 45 facts across 4 types
3. **Single model** - Only GPT-OSS 20B tested

### Suggested Follow-ups

1. **L16E4 Ablation**: What happens when the "fact lookup" expert is removed?
2. **Real Steering**: Implement activation patching at L16
3. **Cross-Model**: Does the L16E4 pattern exist in Llama/Mistral?
4. **Larger Dataset**: TriviaQA, Natural Questions benchmarks
5. **Procedural Strategies**: What enables procedural override?

---
