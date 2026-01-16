# Test-Driven Development

Mantile follows a test-driven development approach. Contributors define expected behavior through tests, and the codebase is designed to support AI-assisted implementation of the necessary components.

## Writing Tests

Tests are written in Markdown format with clear input/output specifications. This makes them readable for both humans and AI agents.

### Test File Structure

Each test file should follow this format:

```markdown
# [Layer Type] Tests

## Test 1: [Description]

### Configuration
- Model: [model_id]
- Hardware: [hardware_id]
- Batch Size: [N]
- Sequence Length: [N]
- Dtype: [bf16/fp16/fp8/int8]
- Parallelism:
  - Tensor Parallel: [N]
  - Context Parallel: [N] (if applicable)
  - Sequence Parallel: [N] (if applicable)

### Expected Behavior
- Memory per chip: ~[X] GB
- Compute time: ~[X] ms
- Communication time: ~[X] ms
- [Other relevant metrics]

### Rationale
[Explanation of why these values are expected, including calculations or references]
```

## Example: Attention Layer Test

See [attention_tests.md](attention_tests.md) for a complete example:

```markdown
## Test 1: Llama 3.3 70B Attention on NVL-72

### Configuration
- Model: meta-llama_Llama-3.3-70B-Instruct
- Hardware: nvidia_nvl72_rack
- Batch Size: 1
- Sequence Length: 2048
- Dtype: bf16
- Parallelism:
  - Tensor Parallel: 8
  - Context Parallel: 1

### Expected Behavior
- Memory per chip: ~45 GB
- Compute time: ~12 ms
- KV cache: ~8 GB

### Rationale
With TP=8 across NVL-72's 72 chips, each chip handles 1/8 of the attention heads...
```

## Running Tests

Currently, tests serve as specification documents. To validate implementations:

1. Run the backend: `./run_backend.sh`
2. Execute the test case using the API
3. Compare actual vs expected results

### Example Validation

```bash
# Run test case
curl -X POST http://localhost:8000/config/system-metrics \
  -H 'Content-Type: application/json' \
  -d @test_payload.json

# Compare output to expected values in test file
```

## Contributing Tests

When contributing new tests:

1. **Be specific**: Include exact configuration values
2. **Show your work**: Explain calculations in the rationale
3. **Use real hardware**: Reference actual accelerator specs
4. **Cover edge cases**: Test unusual parallelism strategies or dtypes
5. **Document assumptions**: Note any simplifications or approximations

### Test Categories

- **Layer tests**: Validate individual layer implementations
- **Parallelism tests**: Verify different parallelism strategies work correctly
- **Hardware tests**: Ensure new hardware configs produce reasonable results
- **Model tests**: Validate new model architectures

## Future: AI-Assisted Implementation

The long-term vision is for AI agents to:
1. Parse test specifications from these Markdown files
2. Validate that implementations satisfy the tests
3. Generate or fix code to match expected behavior

The structured format makes this machine-readable while remaining human-friendly.

---

## Example Test Files

- [attention_tests.md](attention_tests.md) - Attention layer test cases
- [mlp_tests.md](mlp_tests.md) - MLP/Feedforward layer test cases

## Need Help?

- Check existing test files for examples
- Open a GitHub issue with the `test` label
- Reach out to maintainers for guidance
