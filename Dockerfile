FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openmpi-bin \
    fftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy project files
WORKDIR /app
COPY pyproject.toml .
COPY poetry.lock .
COPY . .

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Set environment variables
ENV PYTHONPATH=/app
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default command
CMD ["poetry", "run", "python", "-m", "em_sim"]
