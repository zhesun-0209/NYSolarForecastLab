# Contributing

Contributions should preserve the benchmark contract used by the paper:

- Keep train/validation/test splits, feature definitions, and metrics explicit.
- Avoid model-specific preprocessing unless it is documented and applied consistently.
- Add or update tests when changing data windowing, feature construction, metrics, or model interfaces.
- Keep generated experiment outputs under `results/` rather than committing them.
- Document any new dataset columns in `data/README.md`.

Before opening a pull request, run:

```bash
python run.py --help
python run.py config
python -m pytest -q
```

For changes that affect training behavior, also run:

```bash
python run.py forecast --plant-id 171 --test-mode --test-model Linear --output-dir results/smoke
```
