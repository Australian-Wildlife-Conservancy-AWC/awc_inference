# Publishing to PyPI

## Prerequisites

1. Install build tools:
   ```bash
   pip install build twine
   ```

2. Have your PyPI API token ready

## Windows

```batch
publish.bat [conda_env_name]
```

Example:
```batch
publish.bat awc
```

## Linux / Mac

```bash
chmod +x publish.sh   # First time only
./publish.sh [conda_env_name]
```

Example:
```bash
./publish.sh awc
```

## Notes

- Default conda environment is `awc` if not specified
- You will be prompted for your PyPI API token during upload
- Remember to bump the version in `pyproject.toml` before publishing
