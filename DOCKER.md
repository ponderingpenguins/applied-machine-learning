# Docker Setup for Gait Classification API

This guide explains how to build and run the Gait Classification API using Docker.

## Quick Start


```bash
docker-compose up --build
```

The app will be available at `http://localhost:8000`

## Accessing the App

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc


## Environment Variables

The following can be set in the `docker-compose.yml` file:

- `PYTHONUNBUFFERED=1` — Ensures Python output is streamed directly (recommended)

## Volume Mounts

The docker-compose.yml includes a volume for the checkpoints directory:
```yaml
volumes:
  - ./gait_classification/checkpoints:/app/gait_classification/checkpoints
```

This allows pre-downloaded models to be reused across container restarts without re-downloading.
