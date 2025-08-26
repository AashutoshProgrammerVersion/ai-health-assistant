"""
Authentication Blueprint Initialization

This file creates a Flask Blueprint for authentication-related functionality.
Blueprints organize related routes into separate modules for better code organization.
This blueprint handles user registration, login, and logout functionality.
"""

# FLASK BLUEPRINT IMPORT - For organizing routes into modules (already explained in __init__.py)
from flask import Blueprint

# BLUEPRINT CREATION - Create authentication blueprint
bp = Blueprint('auth', __name__)
"""
'bp = Blueprint('auth', __name__)' - Create a new Flask Blueprint
'bp' - Variable name for our blueprint (short for "blueprint")
'Blueprint()' - Flask class for creating route collections
'auth' - Blueprint name (used in URL generation like url_for('auth.login'))
'__name__' - Current module name (helps Flask find templates and static files)
This blueprint will contain all authentication-related routes (login, register, logout)
"""

# ROUTE IMPORTS - Import route definitions to register them with this blueprint
from app.auth import routes
"""
'from app.auth import routes' - Import the routes module
'app.auth' - Current package (app/auth/)
'routes' - The routes.py file containing route definitions
This import registers all the route functions with this blueprint
Even though we don't use 'routes' directly, importing it activates the routes
"""
