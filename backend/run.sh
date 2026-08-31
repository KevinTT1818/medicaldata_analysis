#!/usr/bin/env bash
cd "$(dirname "$0")"
exec .venv/bin/uvicorn app.main:app --reload --port 8000
