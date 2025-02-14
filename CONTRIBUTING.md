# Contributing to EEMF

## System Dependencies

The project requires these system packages for full functionality:

```bash
sudo apt-get install -y libopenmpi-dev python3-dev
```

## Python Environment Setup

1. Install Python 3.11 (recommended):
```bash
pyenv install 3.11.7
pyenv local 3.11.7
```

2. Install Poetry for dependency management:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install project dependencies:
```bash
poetry install --with mpi
```

## Development Workflow

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and ensure all tests pass:
```bash
poetry run pytest
```

3. Format your code:
```bash
poetry run black .
poetry run isort .
poetry run flake8 .
```

4. Submit a pull request with a clear description of your changes.

## Code Style

- Use Black for code formatting
- Sort imports with isort
- Follow PEP 8 guidelines
- Write docstrings in Google style
