#!/bin/bash

# Render Build Script for AI Health Assistant App
# This script runs during the build phase on Render

echo "Starting build process..."

# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Set environment variables for build
export FLASK_APP=wsgi.py
export FLASK_ENV=production

echo "Build completed successfully!"
