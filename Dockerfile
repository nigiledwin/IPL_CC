# syntax=docker/dockerfile:1

# Use Python 3.11.5 slim image as base
ARG PYTHON_VERSION=3.11.5
FROM python:${PYTHON_VERSION}-slim as base

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Keep Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Create a non-privileged user for the app
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies and install requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Switch to the non-privileged user
USER appuser

# Copy source code into the container
COPY . .

# Expose the port that the application listens on
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py"]
