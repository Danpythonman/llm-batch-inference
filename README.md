# LBI: LLM Batch Inference

Many LLM providers offer 50% token discounts by using batches. This project wraps each provider's batch API in a unified interface.

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
