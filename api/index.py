"""
Vercel API Endpoint for Flask Application

Vercel expects API endpoints in the /api directory.
This file serves as the main entry point for your Flask app on Vercel.
"""

import sys
import os

# Add the parent directory to Python path so we can import our app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from config_production import Config

# Create the Flask application with production config
app = create_app(Config)

# Initialize database tables
with app.app_context():
    db.create_all()

# This function will be called by Vercel
def handler(request):
    return app(request.environ, lambda status, headers: None)

# Export the app for Vercel
application = app

if __name__ == "__main__":
    app.run(debug=False)
