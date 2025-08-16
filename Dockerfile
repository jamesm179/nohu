# Stage 1: Builder stage to install dependencies
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /usr/src/app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements file and install dependencies into the venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final production image
FROM python:3.11-slim

# Set working directory
WORKDIR /home/appuser/app

# Create a non-root user
RUN useradd --create-home appuser
USER appuser

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser . .

# Make the venv's python the default one for the non-root user
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/home/appuser/app"

# Command to run the application
CMD ["python", "main.py"]
