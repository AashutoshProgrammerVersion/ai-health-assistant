"""
Main Blueprint Initialization

This file creates a Flask Blueprint for the main application functionality.
The main blueprint handles core features like the dashboard, health data logging,
and the home page. This separates main functionality from authentication.
"""

# FLASK BLUEPRINT IMPORT - For organizing routes into modules (pattern already explained)
from flask import Blueprint

# BLUEPRINT CREATION - Create main application blueprint
bp = Blueprint('main', __name__)
"""
'bp = Blueprint('main', __name__)' - Create main functionality blueprint
'main' - Blueprint name (used in URL generation like url_for('main.dashboard'))
This blueprint contains the core application features (dashboard, health logging, etc.)
"""

# ROUTE IMPORTS - Import route definitions to register them with this blueprint
from app.main import routes
"""
'from app.main import routes' - Import routes from main module
This registers all main application routes (home, dashboard, health data) with the blueprint
"""
