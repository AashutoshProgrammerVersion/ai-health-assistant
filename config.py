"""
Configuration Settings for Flask Application

This file contains all the configuration settings that Flask needs to run properly.
It handles environment variables, database connections, security settings, and AI service configurations.
Think of this as the "settings panel" for your entire web application.
"""

import os
"""
Provides functions to interact with the operating system
Used here for file paths, environment variables, and directory operations
"""
from os.path import abspath, dirname
from dotenv import load_dotenv
"""
'load_dotenv' - Specific function that reads .env file and loads variables
"""

basedir = abspath(dirname(__file__))
"""
Ensures to get the absolute directory of the current file such as like where the file is stored like how basedir here would be the directory address to the folder level3
"""

load_dotenv(os.path.join(basedir, '.env'))
"""
Loads the environmental variables like stuff that helps with some of the functionality of the app to work such as connecting to APIS like Gemini and Google Calender (it is to basically to use the functionality of Google Calender and Gemini such as to get AI responses from Gemini and get the calender data from Google calender)
"""

# Will basically contain all the configuration settings of this app
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    """
    First choice of assignment for SECRET_KEY is the what this written in ".env" for the secret key which will be used to help set the first priority for a live server if not the second key is used for development purposes 
    This key encrypts user sessions, form tokens, and other sensitive data
    """
    
    # SESSION CONFIGURATION - User session persistence settings
    PERMANENT_SESSION_LIFETIME = 2592000  # 30 days in seconds
    SESSION_PERMANENT = True
    SESSION_TYPE = 'filesystem'
    """
    Session configuration for login persistence:
    - PERMANENT_SESSION_LIFETIME: How long sessions last (30 days)
    - SESSION_PERMANENT: Make sessions survive browser restarts
    - SESSION_TYPE: Store sessions on filesystem (more reliable than memory for production)
    This ensures users stay logged in and can login again after app restarts
    """
    
    # DATABASE CONFIGURATION - Where and how to connect to the database
    database_url = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'app.db')
    
    # Fix for Render.com: Render provides postgres:// but SQLAlchemy needs postgresql://
    if database_url and database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    SQLALCHEMY_DATABASE_URI = database_url
    """
    'SQLALCHEMY_DATABASE_URI' - SQLAlchemy setting that specifies database location
    'os.environ.get('DATABASE_URL')' - Try to get database URL from environment (Render provides this)
    'sqlite:///' - SQLite database URL prefix (three slashes for local file)
    
    IMPORTANT: Render.com provides DATABASE_URL with postgres:// prefix, but SQLAlchemy 1.4+
    requires postgresql:// prefix. The if statement above handles this conversion automatically.
    
    Local development uses SQLite, production on Render uses PostgreSQL.
    """
    
    # SQLALCHEMY OPTIMIZATION - Disable unnecessary feature for better performance
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    """
    'SQLALCHEMY_TRACK_MODIFICATIONS = False' - Disable SQLAlchemy event tracking
    'SQLALCHEMY_TRACK_MODIFICATIONS' - SQLAlchemy setting for object change tracking
    'False' - Boolean value to disable this feature
    This feature tracks every change to database objects but uses extra memory
    Setting to False improves performance since we don't need this tracking
    """
    
    # AI SERVICE CONFIGURATION - Google Gemini API settings
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    """
    'GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')' - Google Gemini API key
    'GEMINI_API_KEY' - Configuration key for Google's Gemini AI service
    'os.environ.get('GEMINI_API_KEY')' - Get API key from environment variables
    This key authenticates our app with Google's Gemini AI service for health advice
    """
    
    # GOOGLE CALENDAR API CONFIGURATION - Calendar integration settings
    GOOGLE_CALENDAR_CLIENT_ID = os.environ.get('GOOGLE_CALENDAR_CLIENT_ID')
    GOOGLE_CALENDAR_CLIENT_SECRET = os.environ.get('GOOGLE_CALENDAR_CLIENT_SECRET')
    # SERVER_NAME = 'localhost:5000'  # Uncomment to force localhost URLs for OAuth
    """
    Google Calendar OAuth2 credentials for calendar integration
    These allow users to connect their Google Calendar for AI schedule optimization
    Note: For OAuth to work, make sure your Google Cloud Console has these redirect URIs:
    - http://127.0.0.1:5000/google_calendar/callback
    - http://localhost:5000/google_calendar/callback
    """
    
    # FILE UPLOAD CONFIGURATION - Health data file processing settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max total upload size
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    ALLOWED_HEALTH_FILE_EXTENSIONS = {
        'csv', 'json', 'txt', 'xml', 'pdf'  # Support multiple wearable data formats
    }
    """
    File upload settings for health data processing
    Supports Samsung Health, Apple Health, Fitbit, Garmin data formats
    Replaces Samsung Health API integration with drag-and-drop file upload
    """
    
    # AI PROCESSING SETTINGS - Enhanced ML model configurations
    SPACY_MODEL = 'en_core_web_sm'  # Lightweight English model for local processing
    GEMINI_MODEL = 'gemini-2.0-flash-exp'  # Latest experimental Gemini model with 32K output tokens
    HEALTH_SCORE_WEIGHTS = {
        'sleep_hours': 0.25,
        'water_intake': 0.15,
        'activity_level': 0.20,
        'heart_rate': 0.15,
        'steps_count': 0.15,
        'mood': 0.10
    }
    """
    Enhanced configuration for AI and health analytics
    SPACY_MODEL: spaCy model for NLP processing of user preferences
    GEMINI_MODEL: Gemini 2.5 Flash for health data processing and recommendations
    HEALTH_SCORE_WEIGHTS: Importance weights for different health metrics
    """
