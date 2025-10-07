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
# Use pip to install the model directly - more reliable on cloud platforms
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Create uploads directory if it doesn't exist
echo "📁 Creating uploads directory..."
mkdir -p uploads

# Run database migrations
echo "🗄️ Running database migrations..."
# First, try to upgrade
flask db upgrade || {
    echo "⚠️  Migration failed, initializing fresh database..."
    # If upgrade fails, initialize from scratch
    python -c "from app import create_app, db; app = create_app(); app.app_context().push(); db.create_all(); print('✅ Database tables created')"
}

echo "✅ Build completed successfully!"
