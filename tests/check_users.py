#!/usr/bin/env python3
"""
Database User Management Script

Check existing users and create the 'jack' user if needed.
"""

import sys
import os

# Add the level3 directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import User

def check_and_create_users():
    """Check existing users and create jack user if needed"""
    app = create_app()
    
    with app.app_context():
        print("🔍 Checking existing users in database...")
        
        # Check all users
        users = User.query.all()
        print(f"📊 Found {len(users)} users in database:")
        
        for user in users:
            print(f"  👤 ID: {user.id}, Username: {user.username}, Email: {user.email}")
        
        # Check if jack user exists
        jack_user = User.query.filter_by(username='jack').first()
        
        if jack_user:
            print(f"\n✅ User 'jack' exists with ID: {jack_user.id}")
            print(f"   📧 Email: {jack_user.email}")
            
            # Test password check
            test_password = input("\n🔐 Enter password to test for 'jack': ")
            if jack_user.check_password(test_password):
                print("✅ Password is correct!")
            else:
                print("❌ Password is incorrect!")
                
                # Offer to reset password
                reset = input("🔄 Would you like to reset jack's password? (y/n): ")
                if reset.lower() == 'y':
                    new_password = input("🔑 Enter new password for jack: ")
                    jack_user.set_password(new_password)
                    db.session.commit()
                    print("✅ Password reset successfully!")
        else:
            print(f"\n❌ User 'jack' does not exist")
            
            # Offer to create jack user
            create = input("🆕 Would you like to create user 'jack'? (y/n): ")
            if create.lower() == 'y':
                email = input("📧 Enter email for jack: ")
                password = input("🔑 Enter password for jack: ")
                
                # Create new user
                new_user = User(username='jack', email=email)
                new_user.set_password(password)
                
                db.session.add(new_user)
                db.session.commit()
                
                print(f"✅ Created user 'jack' with ID: {new_user.id}")
                
        print(f"\n📊 Total users after operation: {User.query.count()}")

if __name__ == '__main__':
    check_and_create_users()