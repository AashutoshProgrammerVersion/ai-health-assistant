#!/usr/bin/env python3
"""
Production WSGI Entry Point for Render Deployment

This file serves as the entry point for the Flask application when deployed
to Render or other WSGI-compatible hosting platforms.
"""

import os
import logging
from app import create_app, db
from app.models import User, HealthData, CalendarEvent, UserPreferences, AIRecommendation

# Create the Flask application instance
app = create_app(os.getenv('FLASK_CONFIG') or 'production')

# Configure logging for production
if not app.debug and not app.testing:
    if app.config['LOG_TO_STDOUT']:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        app.logger.addHandler(stream_handler)
    
    app.logger.setLevel(logging.INFO)
    app.logger.info('AI Health Assistant startup')

# Create database tables if they don't exist
@app.before_first_request
def create_tables():
    """Create database tables on first request"""
    try:
        db.create_all()
        app.logger.info('Database tables created successfully')
    except Exception as e:
        app.logger.error(f'Error creating database tables: {e}')

# Flask shell context for easier debugging
@app.shell_context_processor
def make_shell_context():
    """Add database models to Flask shell context"""
    return {
        'db': db,
        'User': User,
        'HealthData': HealthData,
        'CalendarEvent': CalendarEvent,
        'UserPreferences': UserPreferences,
        'AIRecommendation': AIRecommendation
    }

if __name__ == '__main__':
    # This is for local development only
    # Render will use the app object directly
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
