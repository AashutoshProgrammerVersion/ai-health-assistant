#!/usr/bin/env bash
# exit on error
set -o errexit

echo "🚀 Starting build process..."

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Download spaCy language model (required for NLP features)
echo "🧠 Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Create uploads directory if it doesn't exist
echo "📁 Creating uploads directory..."
mkdir -p uploads

# Run database migrations
echo "🗄️ Running database migrations..."
flask db upgrade

echo "✅ Build completed successfully!"
