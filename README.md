# LBI: LLM Batch Inference

Many LLM providers offer 50% token discounts by using batches. This project wraps each provider's batch API in a unified interface.

## Running Tests

To run the unit tests:

```bash
uv run pytest
```

To run the integration tests:

```bash
uv run pytest -m integration
```

Use the verbose flag `-v` to get more detailed output and use the `-n` flag to run multiple tests in parallel (`-n auto` runs one test per CPU core in parallel).

Example:

```bash
uv run pytest -n auto -v -m integration
```

You can also run specific tests and fixtures by keyword:

```bash
uv run pytest -n auto -v -m integration -k 'full_batch_lifecycle and mistral'
```

## Publishing to PyPI

1. Bump the version in `[pyproject.toml](./pyproject.toml)`:

   ```
   version = "0.1.2"
   ```

2. Build the package:

   ```
   uv build
   ```

3. Publish package:

   ```
   uv publish
   ```

4. Add tag in Git:

   ```
   git add pyproject.toml
   git commit -m "Bump version to 0.1.2"
   git tag v0.1.2
   git push origin main --tags
   ```
