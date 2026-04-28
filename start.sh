#!/usr/bin/env bash

echo "Starting AgroSense AI Backend..."

uvicorn main:app --host 0.0.0.0 --port 10000