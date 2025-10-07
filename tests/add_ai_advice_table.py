#!/usr/bin/env python3
"""
Database Migration: Add PersonalizedHealthAdvice Table

This script adds a new table to store persistent personalized health advice
that stays visible until the user manually refreshes it.

Usage:
    python add_ai_advice_table.py
"""

import os
import sys
from sqlalchemy import inspect

# Add the current directory to the path so we can import our app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db

def create_advice_table():
    """Create the PersonalizedHealthAdvice table"""
    
    app = create_app()
    
    with app.app_context():
        try:
            # Check if table already exists
            inspector = inspect(db.engine)
            if 'personalized_health_advice' in inspector.get_table_names():
                print("✅ PersonalizedHealthAdvice table already exists!")
                return True
            
            # Import the model inside app context to avoid import issues
            from app.models import PersonalizedHealthAdvice
            
            # Create the table
            print("🔨 Creating PersonalizedHealthAdvice table...")
            db.create_all()
            
            print("✅ PersonalizedHealthAdvice table created successfully!")
            print("\nTable structure:")
            print("- id: Primary key")
            print("- user_id: Foreign key to User (unique)")
            print("- insights: JSON text field for insights")
            print("- recommendations: JSON text field for recommendations")
            print("- quick_wins: JSON text field for quick wins")
            print("- concerns: JSON text field for concerns")
            print("- motivation: Text field for motivation message")
            print("- source: Source of advice (gemini_ai, rule_based, etc.)")
            print("- health_score_at_generation: Health score when generated")
            print("- generated_at: Timestamp when advice was generated")
            print("- last_updated: Timestamp when advice was last updated")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating table: {e}")
            import traceback
            traceback.print_exc()
            return False

def verify_table():
    """Verify the table was created correctly"""
    
    app = create_app()
    
    with app.app_context():
        try:
            # Import inside app context
            from app.models import PersonalizedHealthAdvice
            
            # Try to query the table
            count = PersonalizedHealthAdvice.query.count()
            print(f"✅ Table verification successful! Current records: {count}")
            return True
        except Exception as e:
            print(f"❌ Table verification failed: {e}")
            return False

if __name__ == '__main__':
    print("🚀 Starting PersonalizedHealthAdvice table migration...")
    print()
    
    # Create the table
    if create_advice_table():
        print()
        print("🔍 Verifying table creation...")
        if verify_table():
            print()
            print("🎉 Migration completed successfully!")
            print()
            print("The PersonalizedHealthAdvice table is now ready to store persistent health advice.")
            print("Users will now have their health advice persist across page reloads until manually refreshed.")
        else:
            print("❌ Migration verification failed!")
            sys.exit(1)
    else:
        print("❌ Migration failed!")
        sys.exit(1)