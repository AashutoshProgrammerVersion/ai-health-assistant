"""
WSGI Configuration for Production Deployment

This file serves as the entry point for WSGI servers (like Gunicorn) in production.
Vercel uses this file to start your Flask application.
"""

import os
from app import create_app, db
from app.models import User, HealthData, CalendarEvent, UserPreferences, AIRecommendation

# Create Flask application instance
app = create_app()

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

# This is what Vercel will use to run your app
if __name__ == "__main__":
    app.run()