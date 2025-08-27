"""
Production Configuration for Vercel Deployment

This configuration is optimized for production deployment on Vercel.
It includes proper security settings, environment variable handling,
and performance optimizations.
"""

import os
from dotenv import load_dotenv

# Get the base directory
basedir = os.path.abspath(os.path.dirname(__file__))

# Load environment variables
load_dotenv()

class Config:
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key-change-this'
    
    # Database - Use PostgreSQL in production or SQLite as fallback
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL:
        # Fix PostgreSQL URL format if needed
        if DATABASE_URL.startswith('postgres://'):
            DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
        SQLALCHEMY_DATABASE_URI = DATABASE_URL
    else:
        # Fallback to SQLite for development/testing
        SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # AI Services
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    
    # Google Calendar API
    GOOGLE_CALENDAR_CLIENT_ID = os.environ.get('GOOGLE_CALENDAR_CLIENT_ID')
    GOOGLE_CALENDAR_CLIENT_SECRET = os.environ.get('GOOGLE_CALENDAR_CLIENT_SECRET')
    
    # File Upload Settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB for production
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    ALLOWED_HEALTH_FILE_EXTENSIONS = {'csv', 'json', 'txt', 'xml', 'pdf'}
    
    # AI Processing Settings
    SPACY_MODEL = 'en_core_web_sm'
    GEMINI_MODEL = 'gemini-2.5-flash'
    HEALTH_SCORE_WEIGHTS = {
        'sleep_hours': 0.25,
        'water_intake': 0.15,
        'activity_level': 0.20,
        'heart_rate': 0.15,
        'steps_count': 0.15,
        'mood': 0.10
    }
    
    # Production optimizations
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1 year for static files