"""
Application Entry Point and Startup Script

This file serves as the main entry point for the Flask web application.
When you run 'python run.py', this is the first file that executes.
It sets up the Flask app and starts the web server.
"""

# IMPORT STATEMENTS - Bringing in code from other files
from app import create_app, db
import logging
"""
'from app import' - This tells Python to look in the 'app' folder (which contains __init__.py)
'create_app' - A function that creates and configures our Flask application
'db' - The database object that manages our SQLite database connection
The comma separates multiple imports from the same module
"""

from app.models import User, HealthData
"""
'from app.models import' - Look in the app/models.py file
'User, HealthData' - Import our database table classes (models)
These represent the structure of our database tables
We import them here so Flask knows about all database tables when it starts
"""

# LOGGING CONFIGURATION - Suppress progress polling spam
class ProgressFilter(logging.Filter):
    """Custom filter to suppress GET /health_data/progress messages"""
    def filter(self, record):
        # Check if this is a werkzeug log message about health_data/progress
        if hasattr(record, 'getMessage'):
            message = record.getMessage()
            return '/health_data/progress' not in message
        return True

# APPLICATION CREATION - Setting up the Flask web application
app = create_app()
"""
'app = create_app()' - Call the create_app function to build our Flask application
'app' - Variable name that holds our complete web application
'create_app()' - Function call (parentheses execute the function)
This returns a configured Flask application ready to run
"""

# MAIN EXECUTION BLOCK - Code that runs when this file is executed directly
if __name__ == '__main__':
    """
    'if __name__ == '__main__':' - Special Python condition
    '__name__' - Built-in Python variable containing the name of the current module
    '__main__' - Special value when Python runs this file directly (not imported)
    This block only runs when you execute 'python run.py', not when imported
    """
    
    with app.app_context():
        """
        'with app.app_context():' - Context manager for Flask application
        'app.app_context()' - Creates an application context for database operations
        'with' - Python keyword for context managers (automatic cleanup)
        This ensures Flask knows which app we're working with for database operations
        """
        
        db.create_all()
        """
        'db.create_all()' - SQLAlchemy method to create all database tables
        'db' - Our database object imported from app
        'create_all()' - Method that creates tables based on our model definitions
        This creates the 'users' and 'health_data' tables if they don't exist
        """
    
    # Apply filter to suppress progress polling spam
    werkzeug_logger = logging.getLogger('werkzeug')
    progress_filter = ProgressFilter()
    werkzeug_logger.addFilter(progress_filter)
    
    app.run(debug=True)
    """
    'app.run(debug=True)' - Start the Flask development web server
    'app' - Our Flask application object
    'run()' - Method that starts the web server to handle HTTP requests
    'debug=True' - Parameter that enables debug mode for development
    Debug mode provides error details and auto-reloads when code changes
    """
