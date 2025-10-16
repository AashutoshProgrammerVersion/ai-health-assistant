"""Initial database schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2025-10-16 12:00:00.000000

This is the initial migration that creates all tables from scratch.
This should run BEFORE any other migrations.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create all tables from models.py"""
    
    # Check if tables already exist (for existing databases)
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    existing_tables = inspector.get_table_names()
    
    # Create user table if it doesn't exist
    if 'user' not in existing_tables:
        op.create_table('user',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('username', sa.String(length=80), nullable=False),
            sa.Column('email', sa.String(length=120), nullable=False),
            sa.Column('password_hash', sa.String(length=255), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_user_email'), 'user', ['email'], unique=True)
        op.create_index(op.f('ix_user_username'), 'user', ['username'], unique=True)
    
    # Create health_data table if it doesn't exist
    if 'health_data' not in existing_tables:
        op.create_table('health_data',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('date_logged', sa.Date(), nullable=False),
            sa.Column('data_source', sa.String(length=50), nullable=True),
            
            # Activity metrics
            sa.Column('steps', sa.Integer(), nullable=True),
            sa.Column('distance_km', sa.Float(), nullable=True),
            sa.Column('calories_total', sa.Integer(), nullable=True),
            sa.Column('active_minutes', sa.Integer(), nullable=True),
            sa.Column('floors_climbed', sa.Integer(), nullable=True),
            
            # Workout details
            sa.Column('workout_type', sa.String(length=100), nullable=True),
            sa.Column('workout_duration_minutes', sa.Integer(), nullable=True),
            sa.Column('workout_intensity', sa.String(length=50), nullable=True),
            sa.Column('workout_calories', sa.Integer(), nullable=True),
            
            # Heart rate
            sa.Column('heart_rate_avg', sa.Integer(), nullable=True),
            sa.Column('heart_rate_resting', sa.Integer(), nullable=True),
            sa.Column('heart_rate_max', sa.Integer(), nullable=True),
            sa.Column('heart_rate_variability', sa.Float(), nullable=True),
            
            # Sleep
            sa.Column('sleep_duration_hours', sa.Float(), nullable=True),
            sa.Column('sleep_quality_score', sa.Integer(), nullable=True),
            sa.Column('sleep_deep_minutes', sa.Integer(), nullable=True),
            sa.Column('sleep_rem_minutes', sa.Integer(), nullable=True),
            sa.Column('sleep_awake_minutes', sa.Integer(), nullable=True),
            
            # Vitals
            sa.Column('blood_oxygen_percent', sa.Float(), nullable=True),
            sa.Column('body_temperature', sa.Float(), nullable=True),
            sa.Column('systolic_bp', sa.Integer(), nullable=True),
            sa.Column('diastolic_bp', sa.Integer(), nullable=True),
            sa.Column('stress_level', sa.Integer(), nullable=True),
            
            # Nutrition
            sa.Column('water_intake_liters', sa.Float(), nullable=True),
            sa.Column('calories_consumed', sa.Integer(), nullable=True),
            sa.Column('protein_grams', sa.Float(), nullable=True),
            sa.Column('carbs_grams', sa.Float(), nullable=True),
            sa.Column('fat_grams', sa.Float(), nullable=True),
            sa.Column('fiber_grams', sa.Float(), nullable=True),
            
            # Body composition
            sa.Column('weight_kg', sa.Float(), nullable=True),
            sa.Column('body_fat_percent', sa.Float(), nullable=True),
            sa.Column('muscle_mass_kg', sa.Float(), nullable=True),
            sa.Column('bmi', sa.Float(), nullable=True),
            
            # Mental health & lifestyle
            sa.Column('mood_score', sa.Integer(), nullable=True),
            sa.Column('energy_level', sa.Integer(), nullable=True),
            sa.Column('meditation_minutes', sa.Integer(), nullable=True),
            sa.Column('screen_time_hours', sa.Float(), nullable=True),
            sa.Column('social_interactions', sa.Integer(), nullable=True),
            
            # Notes
            sa.Column('notes', sa.Text(), nullable=True),
            
            sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
    
    # Create calendar_event table if it doesn't exist
    if 'calendar_event' not in existing_tables:
        op.create_table('calendar_event',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('title', sa.String(length=200), nullable=False),
            sa.Column('start_time', sa.DateTime(), nullable=False),
            sa.Column('end_time', sa.DateTime(), nullable=False),
            sa.Column('event_type', sa.String(length=50), nullable=True),
            sa.Column('priority', sa.Integer(), nullable=True),
            sa.Column('google_event_id', sa.String(length=200), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
    
    # Create user_preferences table if it doesn't exist
    if 'user_preferences' not in existing_tables:
        op.create_table('user_preferences',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('ai_advice_enabled', sa.Boolean(), nullable=True),
            sa.Column('ai_model_preference', sa.String(length=50), nullable=True),
            sa.Column('notification_frequency', sa.String(length=20), nullable=True),
            sa.Column('health_data_uploaded', sa.Boolean(), nullable=True),
            sa.Column('timezone', sa.String(length=50), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
    
    # Create ai_recommendation table if it doesn't exist
    if 'ai_recommendation' not in existing_tables:
        op.create_table('ai_recommendation',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('recommendation_type', sa.String(length=50), nullable=True),
            sa.Column('title', sa.String(length=200), nullable=True),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('priority', sa.Integer(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('expires_at', sa.DateTime(), nullable=True),
            sa.Column('is_active', sa.Boolean(), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
    
    # Create personalized_health_advice table if it doesn't exist
    if 'personalized_health_advice' not in existing_tables:
        op.create_table('personalized_health_advice',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('insights', sa.Text(), nullable=True),
            sa.Column('recommendations', sa.Text(), nullable=True),
            sa.Column('quick_wins', sa.Text(), nullable=True),
            sa.Column('concerns', sa.Text(), nullable=True),
            sa.Column('motivation', sa.Text(), nullable=True),
            sa.Column('source', sa.String(length=50), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
            sa.PrimaryKeyConstraint('id')
        )


def downgrade():
    """Drop all tables"""
    op.drop_table('personalized_health_advice')
    op.drop_table('ai_recommendation')
    op.drop_table('user_preferences')
    op.drop_table('calendar_event')
    op.drop_table('health_data')
    op.drop_index(op.f('ix_user_username'), table_name='user')
    op.drop_index(op.f('ix_user_email'), table_name='user')
    op.drop_table('user')
