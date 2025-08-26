"""
Authentication Routes for User Login, Registration, and Logout

This file contains Flask route functions that handle user authentication.
Routes are functions that respond to specific URLs and HTTP methods.
These routes process login/register forms and manage user sessions.
"""

# FLASK CORE IMPORTS - Essential Flask functionality for routes
from flask import render_template, redirect, url_for, flash, request
"""
'from flask import render_template, redirect, url_for, flash, request' - Flask utilities
'render_template' - Function to render HTML templates with data
'redirect' - Function to send user to a different URL
'url_for' - Function to generate URLs for routes by name
'flash' - Function to display temporary messages to users
'request' - Object containing information about the current HTTP request
"""

# FLASK-LOGIN IMPORTS - User authentication functionality
from flask_login import current_user, login_user, logout_user
"""
'from flask_login import current_user, login_user, logout_user' - Authentication functions
'current_user' - Object representing the currently logged-in user
'login_user' - Function to log a user into the session
'logout_user' - Function to log out the current user
"""

# URL PARSING IMPORT - Security for redirect URLs
from werkzeug.urls import url_parse
"""
'from werkzeug.urls import url_parse' - URL parsing utility
'werkzeug.urls' - Werkzeug's URL handling module
'url_parse' - Function to safely parse and validate URLs
Used to prevent malicious redirects to external sites
"""

# APPLICATION IMPORTS - Database and blueprint from our app
from app import db
"""
'from app import db' - Import database instance (already explained in models.py)
"""

from app.auth import bp
"""
'from app.auth import bp' - Import authentication blueprint
'bp' - The blueprint created in app/auth/__init__.py
Used with @bp.route decorator to register routes with this blueprint
"""

from app.models import User
"""
'from app.models import User' - Import User model (already explained in models.py)
"""

from app.auth.forms import LoginForm, RegistrationForm
"""
'from app.auth.forms import LoginForm, RegistrationForm' - Import form classes
'LoginForm, RegistrationForm' - Form classes defined in forms.py
Used to create and validate login and registration forms
"""

# LOGIN ROUTE - Handle user authentication
@bp.route('/login', methods=['GET', 'POST'])
def login():
    """
    '@bp.route('/login', methods=['GET', 'POST'])' - Route decorator
    '@bp.route()' - Decorator that registers this function with the blueprint
    '/login' - URL path (becomes /auth/login due to blueprint prefix)
    'methods=['GET', 'POST']' - HTTP methods this route accepts
    'GET' - Used when user visits the page (show login form)
    'POST' - Used when user submits the form (process login)
    
    'def login():' - Function that handles login requests
    'login' - Function name (also used in url_for('auth.login'))
    """
    
    # CHECK IF USER IS ALREADY LOGGED IN
    if current_user.is_authenticated:
        """
        'if current_user.is_authenticated:' - Check if user is already logged in
        'current_user' - Flask-Login object representing current user
        'is_authenticated' - Property that returns True if user is logged in
        """
        return redirect(url_for('main.dashboard'))
        """
        'return redirect(url_for('main.dashboard'))' - Redirect to dashboard
        'redirect()' - Flask function to send user to different URL
        'url_for('main.dashboard')' - Generate URL for dashboard route
        'main.dashboard' - Blueprint.route combination
        If user is already logged in, send them to their dashboard instead
        """
    
    # CREATE AND PROCESS LOGIN FORM
    form = LoginForm()
    """
    'form = LoginForm()' - Create instance of login form
    'form' - Variable holding the form object
    'LoginForm()' - Create new instance of LoginForm class from forms.py
    """
    
    if form.validate_on_submit():
        """
        'if form.validate_on_submit():' - Check if form was submitted and is valid
        'form.validate_on_submit()' - WTForms method that checks:
        1. Was form submitted via POST?
        2. Are all validation rules passed?
        3. Is CSRF token valid?
        Returns True only if all checks pass
        """
        
        # FIND USER IN DATABASE
        user = User.query.filter_by(username=form.username.data).first()
        """
        'user = User.query.filter_by(username=form.username.data).first()' - Find user
        'User.query' - SQLAlchemy query object for User table
        'filter_by(username=form.username.data)' - Find user with matching username
        'form.username.data' - The username entered in the form
        'first()' - Get first matching user, or None if not found
        """
        
        # VALIDATE USER AND PASSWORD
        if user is None or not user.check_password(form.password.data):
            """
            'if user is None or not user.check_password(form.password.data):' - Validate credentials
            'user is None' - True if no user found with that username
            'or' - Logical OR operator (either condition makes this True)
            'not user.check_password(form.password.data)' - True if password is wrong
            'user.check_password()' - Method from User model to verify password
            'form.password.data' - The password entered in the form
            """
            flash('Invalid username or password', 'danger')
            """
            'flash('Invalid username or password', 'danger')' - Show error message
            'flash()' - Flask function to show temporary message to user
            'Invalid username or password' - Error message text
            'danger' - Bootstrap CSS class for red error styling
            """
            return redirect(url_for('auth.login'))
            """
            'return redirect(url_for('auth.login'))' - Redirect back to login page
            Shows the form again with the error message displayed
            """
        
        # LOG USER IN
        login_user(user, remember=form.remember_me.data)
        """
        'login_user(user, remember=form.remember_me.data)' - Log user into session
        'login_user()' - Flask-Login function to create user session
        'user' - The User object to log in
        'remember=form.remember_me.data' - Whether to remember login across browser restarts
        'form.remember_me.data' - Value of "Remember Me" checkbox
        """
        
        # HANDLE REDIRECT AFTER LOGIN
        next_page = request.args.get('next')
        """
        'next_page = request.args.get('next')' - Get redirect destination
        'request.args' - Dictionary of URL parameters
        'get('next')' - Get value of 'next' parameter, or None if not present
        'next' parameter is added when user is redirected to login from protected page
        """
        
        if not next_page or url_parse(next_page).netloc != '':
            """
            'if not next_page or url_parse(next_page).netloc != '':' - Validate redirect URL
            'not next_page' - True if no 'next' parameter was provided
            'url_parse(next_page).netloc' - Extract domain from URL
            'netloc != ''' - True if URL contains a domain (external site)
            This prevents malicious redirects to external websites
            """
            next_page = url_for('main.dashboard')
            """
            'next_page = url_for('main.dashboard')' - Default redirect destination
            If no safe redirect URL, send user to dashboard
            """
        return redirect(next_page)
        """
        'return redirect(next_page)' - Redirect to final destination
        Either the originally requested page or the dashboard
        """
    
    # SHOW LOGIN FORM (for GET requests or invalid submissions)
    return render_template('auth/login.html', title='Sign In', form=form)
    """
    'return render_template('auth/login.html', title='Sign In', form=form)' - Render login page
    'render_template()' - Flask function to render HTML template
    'auth/login.html' - Template file path (in templates/auth/ directory)
    'title='Sign In'' - Variable passed to template for page title
    'form=form' - Pass form object to template for rendering HTML form
    """

# LOGOUT ROUTE - Handle user logout
@bp.route('/logout')
def logout():
    """
    '@bp.route('/logout')' - Logout route decorator
    '/logout' - URL path (becomes /auth/logout)
    No methods specified, defaults to GET only
    
    'def logout():' - Function to handle logout
    """
    logout_user()
    """
    'logout_user()' - Flask-Login function to end user session
    Clears all session data related to the logged-in user
    """
    return redirect(url_for('main.index'))
    """
    'return redirect(url_for('main.index'))' - Redirect to home page
    'main.index' - Home page route
    After logout, send user to the public home page
    """

# REGISTRATION ROUTE - Handle new user account creation
@bp.route('/register', methods=['GET', 'POST'])
def register():
    """
    '@bp.route('/register', methods=['GET', 'POST'])' - Registration route
    '/register' - URL path (becomes /auth/register)
    'GET' - Show registration form
    'POST' - Process registration form
    """
    
    # CHECK IF USER IS ALREADY LOGGED IN
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
        """
        Same pattern as login route - redirect logged-in users to dashboard
        """
    
    # CREATE AND PROCESS REGISTRATION FORM
    form = RegistrationForm()
    """
    'form = RegistrationForm()' - Create registration form instance
    'RegistrationForm()' - Form class from forms.py with username, email, password fields
    """
    
    if form.validate_on_submit():
        """
        'if form.validate_on_submit():' - Check if registration form is valid
        Same validation process as login form
        Includes custom validation methods from RegistrationForm class
        """
        
        # CREATE NEW USER OBJECT
        user = User(username=form.username.data, email=form.email.data)
        """
        'user = User(username=form.username.data, email=form.email.data)' - Create User object
        'User()' - Create new instance of User model
        'username=form.username.data' - Set username from form
        'email=form.email.data' - Set email from form
        This creates a User object but doesn't save it to database yet
        """
        
        # SET USER PASSWORD
        user.set_password(form.password.data)
        """
        'user.set_password(form.password.data)' - Set encrypted password
        'user.set_password()' - Method from User model that hashes password
        'form.password.data' - Plain text password from form
        This encrypts the password and stores the hash in user.password_hash
        """
        
        # SAVE USER TO DATABASE
        db.session.add(user)
        """
        'db.session.add(user)' - Add user to database session
        'db.session' - SQLAlchemy session for database transactions
        'add(user)' - Stage the user object for database insertion
        This prepares the user for saving but doesn't save yet
        """
        
        db.session.commit()
        """
        'db.session.commit()' - Save all changes to database
        'commit()' - SQLAlchemy method to execute all staged changes
        This actually saves the new user to the database
        """
        
        # SHOW SUCCESS MESSAGE AND REDIRECT
        flash('Congratulations, you are now registered!', 'success')
        """
        'flash('Congratulations, you are now registered!', 'success')' - Success message
        'success' - Bootstrap CSS class for green success styling
        """
        return redirect(url_for('auth.login'))
        """
        'return redirect(url_for('auth.login'))' - Redirect to login page
        After successful registration, user needs to log in with new account
        """
    
    # SHOW REGISTRATION FORM (for GET requests or invalid submissions)
    return render_template('auth/register.html', title='Register', form=form)
    """
    'return render_template('auth/register.html', title='Register', form=form)' - Render registration page
    'auth/register.html' - Registration template file
    'title='Register'' - Page title for registration
    'form=form' - Pass form object to template for HTML generation
    """
