#!/usr/bin/env bash
# exit on error
set -o errexit

echo "🚀 Starting build process..."
echo "⚠️  NOTE: Free tier has 512MB RAM - large file uploads may timeout"
echo "💡 TIP: Upload smaller batches (5-10 files) to avoid memory issues"
echo ""

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
# Run migrations - DO NOT fallback to db.create_all() as it wipes existing data
flask db upgrade

# If migrations fail, the build should fail to prevent data loss
# Check if migrations directory exists
if [ ! -d "migrations/versions" ] || [ -z "$(ls -A migrations/versions)" ]; then
    echo "⚠️  WARNING: No migration files found!"
    echo "💡 Run 'flask db init' and 'flask db migrate' locally first"
    exit 1
fi

echo "✅ Build completed successfully!"
