web: python wsgi.py
release: python -c "from app import create_app, db; app = create_app('production'); app.app_context().push(); db.create_all()"
