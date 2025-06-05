from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, CollectionInvalid
from typing import List, Dict, Any
from config.settings import settings

class MongoDBManager:
    """
    Manges connections and operations for MongoDB
    Adheres to SRP by handling only database interactions.
    """
    def __init__(self, uri: str = settings.MONGODB_URI, db_name: str = settings.MONGODB_DB_NAME):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
    
    def connect(self):
        """Establishes a connection to MongoDB."""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
        
            # Ping
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            print(f"Successfully connected to MongoDB: {self.uri}")
        
        except ConnectionFailure as e:
            print(f"MongoDB connection failed: {e}")
            raise ConnectionFailure(f"Could not connect to MongoDB at {self.uri}. Is it running?") from e
        except Exception as e:
            print(f"An unexpected error occurred during MongoDB connection: {e}")
            raise
    
    def close(self):
        """Closes the MongoDB connection.."""
        if self.client:
            self.client.close()
            print(f"MongoDB connection closed.")
    
    def get_collection(self, collection_name: str):
        """Returns a MongoDB collection object."""
        if self.db is None:
            raise ConnectionFailure("MongoDB not connected. Call connect() first.")
        return self.db[collection_name]
    
    def insert_documents(self, collection_name: str, documents: List[Dict[str, Any]], drop_existing = False):
        """
        Inserts a list of documents into a specified collection.
        Optionally drops the collection before insertion.
        """
        if self.db is None:
            raise ConnectionFailure("MongoDB not connected. Call connect() first.")
        
        collection = self.db[collection_name]
        if drop_existing:
            try:
                collection.drop()
                print(f"Dropped collection: {collection_name}")
            except CollectionInvalid:
                print(f"Collection {collection_name} did not exist, so not dropped.")
            except Exception as e:
                print(f"Error dropping collection {collection_name}: {e}")
        
        if documents:
            result = collection.insert_many(documents)
            print(f"Inserted {len(result.inserted_ids)} documents into '{collection_name}' collection.")
        else:
            print(f"No documents to insert into '{collection_name}'.")
    
    def find_documents(self, collection_name: str, query: Dict[str, Any] = None, limit: int = 0) -> List[Dict[str, Any]]:
        """
        Finds documents in a specified collection based on a query.
        """
        if self.db is None:
            raise ConnectionFailure("MongoDB not connected. Call connect() first.")

        collection = self.db[collection_name]
        if query is None:
            query = {}
            
        results = []
        cursor = collection.find(query)
        if limit > 0:
            cursor = cursor.limit(limit)
            
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            results.append(doc)
            
        return results
        