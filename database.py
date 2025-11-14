# database.py
"""
MongoDB Database Configuration and Helper Functions
"""
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import os
from datetime import datetime

class Database:
    """MongoDB Database Handler"""
    
    def __init__(self):
        self.client = None
        self.db = None
        
    def connect(self, connection_string=None):
        """
        Connect to MongoDB
        
        Args:
            connection_string: MongoDB connection string. 
                             If None, uses environment variable or default localhost
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Get connection string from parameter, env variable, or use default
            if connection_string is None:
                connection_string = os.getenv(
                    'MONGODB_URI', 
                    'mongodb://127.0.0.1:27017/'
                )
            
            # Create MongoDB client
            self.client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000  # 5 second timeout
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Connect to database (you can change 'thilak' to your preferred DB name)
            db_name = os.getenv('MONGODB_DB_NAME', 'thilak')
            self.db = self.client[db_name]
            
            print(f"✓ Successfully connected to MongoDB database: {db_name}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"✗ Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error connecting to MongoDB: {e}")
            return False
    
    def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("✓ MongoDB connection closed")
    
    # ==================== USER MANAGEMENT ====================
    
    def create_user(self, username, password_hash, email=None):
        """
        Create a new user
        
        Args:
            username: User's username
            password_hash: Hashed password (use werkzeug.security.generate_password_hash)
            email: User's email (optional)
        
        Returns:
            dict: Created user document or None if failed
        """
        try:
            users_collection = self.db['users']
            
            # Check if user already exists
            if users_collection.find_one({'username': username}):
                print(f"✗ User '{username}' already exists")
                return None
            
            user_doc = {
                'username': username,
                'password_hash': password_hash,
                'email': email,
                'created_at': datetime.utcnow(),
                'last_login': None
            }
            
            result = users_collection.insert_one(user_doc)
            user_doc['_id'] = result.inserted_id
            
            print(f"✓ User '{username}' created successfully")
            return user_doc
            
        except Exception as e:
            print(f"✗ Error creating user: {e}")
            return None
    
    def find_user(self, username):
        """
        Find user by username
        
        Args:
            username: Username to search for
        
        Returns:
            dict: User document or None if not found
        """
        try:
            users_collection = self.db['users']
            return users_collection.find_one({'username': username})
        except Exception as e:
            print(f"✗ Error finding user: {e}")
            return None
    
    def update_last_login(self, username):
        """Update user's last login timestamp"""
        try:
            users_collection = self.db['users']
            users_collection.update_one(
                {'username': username},
                {'$set': {'last_login': datetime.utcnow()}}
            )
        except Exception as e:
            print(f"✗ Error updating last login: {e}")
    
    # ==================== ANALYSIS RECORDS ====================
    
    def save_analysis(self, username, patient_name, patient_age, analysis_data):
        """
        Save analysis results to database
        
        Args:
            username: Username of the user performing analysis
            patient_name: Patient's name
            patient_age: Patient's age
            analysis_data: Dictionary containing analysis results
        
        Returns:
            str: Analysis ID or None if failed
        """
        try:
            analyses_collection = self.db['analyses']
            
            analysis_doc = {
                'username': username,
                'patient_name': patient_name,
                'patient_age': patient_age,
                'analysis_datetime': datetime.utcnow(),
                'original_image_path': analysis_data.get('original_image_path'),
                'filtered_image_path': analysis_data.get('filtered_image_path'),
                'denoised_image_path': analysis_data.get('denoised_image_path'),
                'ensemble_prediction': analysis_data.get('ensemble_prediction'),
                'ensemble_confidence': analysis_data.get('ensemble_confidence'),
                'individual_results': analysis_data.get('individual_results'),
                'comprehensive_report': analysis_data.get('comprehensive_report'),
                'synthetic_report': analysis_data.get('synthetic_report'),
                'gradcam_paths': analysis_data.get('gradcam_paths', [])
            }
            
            result = analyses_collection.insert_one(analysis_doc)
            print(f"✓ Analysis saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"✗ Error saving analysis: {e}")
            return None
    
    def get_user_analyses(self, username, limit=10):
        """
        Get recent analyses for a user
        
        Args:
            username: Username to fetch analyses for
            limit: Maximum number of analyses to return
        
        Returns:
            list: List of analysis documents
        """
        try:
            analyses_collection = self.db['analyses']
            analyses = list(
                analyses_collection.find({'username': username})
                .sort('analysis_datetime', -1)
                .limit(limit)
            )
            return analyses
        except Exception as e:
            print(f"✗ Error fetching analyses: {e}")
            return []
    
    def get_analysis_by_id(self, analysis_id):
        """Get specific analysis by ID"""
        try:
            from bson.objectid import ObjectId
            analyses_collection = self.db['analyses']
            return analyses_collection.find_one({'_id': ObjectId(analysis_id)})
        except Exception as e:
            print(f"✗ Error fetching analysis: {e}")
            return None
    
    # ==================== STATISTICS ====================
    
    def get_user_statistics(self, username):
        """
        Get statistics for a user
        
        Returns:
            dict: Statistics including total analyses, stage distribution, etc.
        """
        try:
            analyses_collection = self.db['analyses']
            
            total_analyses = analyses_collection.count_documents({'username': username})
            
            # Get stage distribution
            pipeline = [
                {'$match': {'username': username}},
                {'$group': {
                    '_id': '$ensemble_prediction',
                    'count': {'$sum': 1}
                }}
            ]
            stage_distribution = list(analyses_collection.aggregate(pipeline))
            
            return {
                'total_analyses': total_analyses,
                'stage_distribution': stage_distribution
            }
        except Exception as e:
            print(f"✗ Error fetching statistics: {e}")
            return None


# Create global database instance
db = Database()


def init_db(app):
    """
    Initialize database connection with Flask app
    
    Args:
        app: Flask application instance
    """
    connection_string = app.config.get('MONGODB_URI')
    if db.connect(connection_string):
        print("✓ Database initialized successfully")
        return True
    else:
        print("✗ Database initialization failed")
        return False


def close_db():
    """Close database connection"""
    db.disconnect()
    print("✓ Database connection closed")


