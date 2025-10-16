"""Initial schema - create all base tables

Revision ID: 001_initial_schema
Revises: 
Create Date: 2025-10-16 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime


# revision identifiers, used by Alembic.
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create user table
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

    # Create health_data table with ALL base fields
    op.create_table('health_data',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('date_logged', sa.Date(), nullable=False),
    # Activity metrics
    sa.Column('steps', sa.Integer(), nullable=True),
    sa.Column('distance_km', sa.Float(), nullable=True),
    sa.Column('calories_total', sa.Integer(), nullable=True),
    sa.Column('active_minutes', sa.Integer(), nullable=True),
    sa.Column('floors_climbed', sa.Integer(), nullable=True),
    # Heart rate metrics
    sa.Column('heart_rate_avg', sa.Integer(), nullable=True),
    sa.Column('heart_rate_resting', sa.Integer(), nullable=True),
    sa.Column('heart_rate_max', sa.Integer(), nullable=True),
    sa.Column('heart_rate_variability', sa.Float(), nullable=True),
    # Sleep metrics
    sa.Column('sleep_duration_hours', sa.Float(), nullable=True),
    sa.Column('sleep_quality_score', sa.Integer(), nullable=True),
    sa.Column('sleep_deep_minutes', sa.Integer(), nullable=True),
    sa.Column('sleep_light_minutes', sa.Integer(), nullable=True),
    sa.Column('sleep_rem_minutes', sa.Integer(), nullable=True),
    sa.Column('sleep_awake_minutes', sa.Integer(), nullable=True),
    # Advanced health metrics
    sa.Column('blood_oxygen_percent', sa.Integer(), nullable=True),
    sa.Column('stress_level', sa.Integer(), nullable=True),
    sa.Column('body_temperature', sa.Float(), nullable=True),
    # Body composition
    sa.Column('weight_kg', sa.Float(), nullable=True),
    sa.Column('body_fat_percent', sa.Float(), nullable=True),
    sa.Column('muscle_mass_kg', sa.Float(), nullable=True),
    # Hydration
    sa.Column('water_intake_liters', sa.Float(), nullable=True),
    # Nutrition metrics
    sa.Column('calories_consumed', sa.Integer(), nullable=True),
    sa.Column('protein_grams', sa.Float(), nullable=True),
    sa.Column('carbs_grams', sa.Float(), nullable=True),
    sa.Column('fat_grams', sa.Float(), nullable=True),
    sa.Column('fiber_grams', sa.Float(), nullable=True),
    # Blood pressure
    sa.Column('systolic_bp', sa.Integer(), nullable=True),
    sa.Column('diastolic_bp', sa.Integer(), nullable=True),
    # BMI
    sa.Column('bmi', sa.Float(), nullable=True),
    # Exercise details
    sa.Column('workout_type', sa.String(length=50), nullable=True),
    sa.Column('workout_duration_minutes', sa.Integer(), nullable=True),
    sa.Column('workout_intensity', sa.String(length=20), nullable=True),
    sa.Column('workout_calories', sa.Integer(), nullable=True),
    # Subjective metrics
    sa.Column('mood_score', sa.Integer(), nullable=True),
    sa.Column('energy_level', sa.Integer(), nullable=True),
    # Lifestyle metrics
    sa.Column('meditation_minutes', sa.Integer(), nullable=True),
    sa.Column('screen_time_hours', sa.Float(), nullable=True),
    sa.Column('social_interactions', sa.Integer(), nullable=True),
    # Notes
    sa.Column('notes', sa.Text(), nullable=True),
    # File processing metadata
    sa.Column('data_source', sa.String(length=100), nullable=True),
    sa.Column('processed_data', sa.Text(), nullable=True),
    sa.Column('extraction_date', sa.DateTime(), nullable=True),
    sa.Column('health_score', sa.Float(), nullable=True),
    sa.Column('device_type', sa.String(length=100), nullable=True),
    # Timestamps
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )

    # Create calendar_event table
    op.create_table('calendar_event',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('title', sa.String(length=200), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('start_time', sa.DateTime(), nullable=False),
    sa.Column('end_time', sa.DateTime(), nullable=False),
    sa.Column('is_ai_modifiable', sa.Boolean(), nullable=True),
    sa.Column('is_fixed_time', sa.Boolean(), nullable=True),
    sa.Column('priority_level', sa.Integer(), nullable=True),
    sa.Column('event_type', sa.String(length=50), nullable=True),
    sa.Column('google_calendar_id', sa.String(length=255), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )

    # Create user_preferences table
    op.create_table('user_preferences',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('ai_optimization_enabled', sa.Boolean(), nullable=True),
    sa.Column('reminder_frequency', sa.String(length=20), nullable=True),
    sa.Column('daily_water_goal', sa.Integer(), nullable=True),
    sa.Column('daily_sleep_goal', sa.Float(), nullable=True),
    sa.Column('daily_steps_goal', sa.Integer(), nullable=True),
    sa.Column('daily_activity_goal', sa.Integer(), nullable=True),
    sa.Column('reminder_water', sa.Boolean(), nullable=True),
    sa.Column('reminder_exercise', sa.Boolean(), nullable=True),
    sa.Column('reminder_sleep', sa.Boolean(), nullable=True),
    sa.Column('reminder_medication', sa.Boolean(), nullable=True),
    sa.Column('reminder_meal', sa.Boolean(), nullable=True),
    sa.Column('reminder_mindfulness', sa.Boolean(), nullable=True),
    sa.Column('smart_reminders_enabled', sa.Boolean(), nullable=True),
    sa.Column('quiet_hours_start', sa.Time(), nullable=True),
    sa.Column('quiet_hours_end', sa.Time(), nullable=True),
    sa.Column('google_calendar_connected', sa.Boolean(), nullable=True),
    sa.Column('health_data_uploaded', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id')
    )

    # Create ai_recommendation table
    op.create_table('ai_recommendation',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('recommendation_type', sa.String(length=50), nullable=False),
    sa.Column('title', sa.String(length=200), nullable=False),
    sa.Column('description', sa.Text(), nullable=False),
    sa.Column('ai_confidence', sa.Float(), nullable=True),
    sa.Column('user_accepted', sa.Boolean(), nullable=True),
    sa.Column('user_feedback', sa.Text(), nullable=True),
    sa.Column('is_implemented', sa.Boolean(), nullable=True),
    sa.Column('implementation_date', sa.DateTime(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )

    # Create personalized_health_advice table
    op.create_table('personalized_health_advice',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('insights', sa.Text(), nullable=True),
    sa.Column('recommendations', sa.Text(), nullable=True),
    sa.Column('quick_wins', sa.Text(), nullable=True),
    sa.Column('concerns', sa.Text(), nullable=True),
    sa.Column('motivation', sa.Text(), nullable=True),
    sa.Column('source', sa.String(length=50), nullable=True),
    sa.Column('health_score_at_generation', sa.Float(), nullable=True),
    sa.Column('generated_at', sa.DateTime(), nullable=True),
    sa.Column('last_updated', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id')
    )

    # Create event_backup table
    op.create_table('event_backup',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('event_id', sa.Integer(), nullable=False),
    sa.Column('original_start_time', sa.DateTime(), nullable=False),
    sa.Column('original_end_time', sa.DateTime(), nullable=False),
    sa.Column('optimization_date', sa.Date(), nullable=False),
    sa.Column('backup_reason', sa.String(length=100), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['event_id'], ['calendar_event.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade():
    op.drop_table('event_backup')
    op.drop_table('personalized_health_advice')
    op.drop_table('ai_recommendation')
    op.drop_table('user_preferences')
    op.drop_table('calendar_event')
    op.drop_table('health_data')
    op.drop_index(op.f('ix_user_username'), table_name='user')
    op.drop_index(op.f('ix_user_email'), table_name='user')
    op.drop_table('user')
