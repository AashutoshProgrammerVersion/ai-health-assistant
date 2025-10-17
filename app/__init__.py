"""
Flask Application Factory and Extension Setup

This file creates and configures the Flask application using the "Application Factory" pattern.
It sets up database connections, user authentication, and organizes the app into blueprints.
This is the central configuration hub that brings together all parts of the application.
"""

# PYTHON STANDARD LIBRARY IMPORTS
import os
import logging

# FLASK CORE IMPORTS - Essential Flask components
from flask import Flask
"""
'from flask import Flask' - Import the main Flask class
'Flask' - The core class that represents a Flask web application
This class handles HTTP requests, routing, and response generation
"""

# DATABASE EXTENSION IMPORT - SQLAlchemy for database operations
from flask_sqlalchemy import SQLAlchemy
"""
'from flask_sqlalchemy import SQLAlchemy' - Import database extension
'flask_sqlalchemy' - Flask extension that integrates SQLAlchemy ORM
'SQLAlchemy' - Class that provides database functionality to Flask
This handles database connections, table creation, and ORM operations
"""

# AUTHENTICATION EXTENSION IMPORT - User login management
from flask_login import LoginManager
"""
'from flask_login import LoginManager' - Import user authentication manager
'flask_login' - Flask extension for handling user sessions and authentication
'LoginManager' - Class that manages user login/logout and session handling
This tracks which users are logged in and protects routes that require authentication
"""

# DATABASE MIGRATION EXTENSION IMPORT - Database schema changes
from flask_migrate import Migrate
"""
'from flask_migrate import Migrate' - Import database migration manager
'flask_migrate' - Flask extension for handling database schema changes
'Migrate' - Class that manages database migrations and upgrades
This handles database schema evolution and version control
"""

# CONFIGURATION IMPORT - Application settings
from config import Config
"""
'from config import Config' - Import our configuration class
'config' - Reference to our config.py file in the project root
'Config' - The configuration class containing all Flask settings
This provides database URLs, secret keys, and other application settings
"""

# GLOBAL EXTENSION INSTANCES - Create extension objects before app creation
db = SQLAlchemy()
"""
'db = SQLAlchemy()' - Create a global SQLAlchemy database instance
'db' - Variable name that will be used throughout the app for database operations
'SQLAlchemy()' - Create an instance of the SQLAlchemy class
This object will handle all database operations (queries, table creation, etc.)
Created globally so it can be imported and used in other files
"""

login = LoginManager()
"""
'login = LoginManager()' - Create a global LoginManager instance
'login' - Variable name for the authentication manager
'LoginManager()' - Create an instance of the LoginManager class
This object handles user authentication, login sessions, and access control
"""

migrate = Migrate()
"""
'migrate = Migrate()' - Create a global Migrate instance
'migrate' - Variable name for the database migration manager
'Migrate()' - Create an instance of the Migrate class
This object handles database schema changes and version control
"""

# LOGIN MANAGER CONFIGURATION - Set up authentication behavior
login.login_view = 'auth.login'
"""
'login.login_view = 'auth.login'' - Configure where to redirect unauthorized users
'login_view' - LoginManager attribute specifying the login route
'auth.login' - Blueprint.route combination (auth blueprint, login route)
When users try to access protected pages without logging in, redirect them here
"""

login.login_message = 'Please log in to access this page.'
"""
'login.login_message = 'Please log in to access this page.'' - Set login prompt message
'login_message' - LoginManager attribute for the message shown to users
'Please log in to access this page.' - User-friendly message explaining why they were redirected
This message appears when users are redirected to login page
"""

# APPLICATION FACTORY FUNCTION - Creates and configures Flask application
def create_app(config_class=Config):
    """
    'def create_app(config_class=Config):' - Define function to create Flask app
    'def' - Python keyword to define a function
    'create_app' - Function name (descriptive of what it does)
    '(config_class=Config)' - Parameter with default value
    'config_class' - Parameter name for configuration settings
    '=Config' - Default parameter value (uses our Config class if none provided)
    This function creates a new Flask application with all extensions configured
    """
    
    # FLASK APPLICATION CREATION - Create the core Flask app object
    app = Flask(__name__)
    """
    'app = Flask(__name__)' - Create a Flask application instance
    'app' - Variable to hold our Flask application
    'Flask(__name__)' - Create Flask app with current module name
    '__name__' - Special Python variable containing current module name
    Flask uses this to find templates, static files, and determine app location
    """
    
    # LOGGING CONFIGURATION - Set up logging to suppress Google API warnings
    logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
    """
    Configure logging to suppress Google API discovery cache warnings
    This prevents the oauth2client<4.0.0 warning from appearing in logs
    """
    
    # CONFIGURATION APPLICATION - Apply settings to the Flask app
    app.config.from_object(config_class)
    """
    'app.config.from_object(config_class)' - Load configuration from our Config class
    'app.config' - Flask's configuration object (dictionary-like)
    'from_object()' - Method to load settings from a class
    'config_class' - Our Config class containing SECRET_KEY, database settings, etc.
    This applies all the settings from config.py to our Flask application
    """
    
    # EXTENSION INITIALIZATION - Connect extensions to the Flask app
    db.init_app(app)
    """
    'db.init_app(app)' - Initialize SQLAlchemy database with our Flask app
    'db' - Our global SQLAlchemy instance created earlier
    'init_app()' - Method that connects the database to a specific Flask app
    'app' - Our Flask application instance
    This tells SQLAlchemy which Flask app to use for database operations
    """
    
    login.init_app(app)
    """
    'login.init_app(app)' - Initialize LoginManager with our Flask app
    'login' - Our global LoginManager instance created earlier
    'init_app()' - Method that connects authentication to a specific Flask app
    This enables user session management and login functionality
    """
    
    migrate.init_app(app, db)
    """
    'migrate.init_app(app, db)' - Initialize Migrate with our Flask app and database
    'migrate' - Our global Migrate instance created earlier
    'init_app()' - Method that connects migration functionality to Flask app
    'app, db' - Parameters: Flask app and SQLAlchemy database instance
    This enables database migration commands and schema version control
    """
    
    # CREATE UPLOAD DIRECTORY - Ensure uploads folder exists for health data files
    upload_folder = app.config.get('UPLOAD_FOLDER')
    if upload_folder and not os.path.exists(upload_folder):
        os.makedirs(upload_folder, exist_ok=True)
    
    # BLUEPRINT IMPORTS AND REGISTRATION - Organize app into modules
    from app.main import bp as main_bp
    """
    'from app.main import bp as main_bp' - Import main blueprint
    'from app.main' - Look in the app/main/ directory
    'import bp' - Import the blueprint object named 'bp'
    'as main_bp' - Give it an alias 'main_bp' in this context
    Blueprints organize related routes and views into separate modules
    """
    
    app.register_blueprint(main_bp)
    """
    'app.register_blueprint(main_bp)' - Register the main blueprint with Flask
    'app' - Our Flask application instance
    'register_blueprint()' - Method to add a blueprint to the app
    'main_bp' - The main blueprint containing routes like dashboard, home page
    This makes all routes in the main blueprint available to the Flask app
    """
    
    from app.auth import bp as auth_bp
    """
    'from app.auth import bp as auth_bp' - Import authentication blueprint
    'app.auth' - Look in the app/auth/ directory
    'bp' - The blueprint object containing authentication routes
    'as auth_bp' - Alias for clarity in this context
    This blueprint handles user registration, login, and logout
    """
    
    # SESSION MANAGEMENT - Handle session cleanup and expiration
    @app.before_request
    def make_session_permanent():
        """
        Make sessions permanent on every request to ensure proper expiration
        This ensures Flask respects the PERMANENT_SESSION_LIFETIME setting
        """
        from flask import session
        session.permanent = True
    """
    '@app.before_request' - Decorator that runs function before each request
    'make_session_permanent()' - Function to ensure session uses permanent lifetime
    This guarantees sessions expire after exactly 30 days, not sooner
    Helps maintain login state even after Render app restarts
    """
    
    app.register_blueprint(auth_bp, url_prefix='/auth')
    """
    'app.register_blueprint(auth_bp, url_prefix='/auth')' - Register auth blueprint
    'register_blueprint()' - Method to add blueprint to Flask app
    'auth_bp' - Authentication blueprint with login/register routes
    'url_prefix='/auth'' - All routes in this blueprint start with /auth/
    This means login route becomes /auth/login, register becomes /auth/register
    """
    
    # APPLICATION RETURN - Send back the configured Flask app
    return app
    """
    'return app' - Send the completed Flask application back to caller
    'return' - Python keyword to send a value back from a function
    'app' - Our fully configured Flask application instance
    The caller (run.py) receives this app and can start the web server
    """

# MODEL IMPORTS - Load database models to register them with SQLAlchemy
from app import models
"""
'from app import models' - Import our database models
'app' - Current package (app directory)
'models' - The models.py file containing User and HealthData classes
This import ensures SQLAlchemy knows about all our database tables
Even though we don't use 'models' directly here, importing it registers the tables
"""
