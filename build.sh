#!/bin/bash

# Render Build Script for AI Health Assistant App
# This script runs during the build phase on Render

echo "Starting build process..."

# Upgrade pip to latest version
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify Flask installation
echo "Verifying Flask installation..."
python -c "import flask; print(f'Flask version: {flask.__version__}')"

# Set environment variables for build
export FLASK_APP=wsgi.py
export FLASK_ENV=production

echo "Build completed successfully!"
echo "Installed packages:"
pip list | grep -E "(Flask|gunicorn|psycopg2)"
