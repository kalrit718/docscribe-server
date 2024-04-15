import os
import pymongo

class DatabaseService():  

  def __init__(self):
    mongo_uri = os.environ.get("MONGO_URI")
    mongo_username = os.environ.get("MONGO_USERNAME")
    mongo_password = os.environ.get("MONGO_PASSWORD")
    mongo_appname = os.environ.get("MONGO_APPNAME")

    self._mongo_client = pymongo.MongoClient(f"mongodb+srv://{mongo_username}:{mongo_password}@{mongo_uri}/?retryWrites=true&w=majority&appName={mongo_appname}")
    self._db = self._mongo_client.diagnostics
    self._diagnostics_data_collection = self._db["diagnostics_data"]
    print('Database Service Initialized')

    try:
      self._mongo_client.admin.command('ping')
      print("Successfully connected to MongoDB!")
    except Exception as e:
      print(e)

  def insert_record(self, username, data):
    inserted_result = self._diagnostics_data_collection.insert_one({"username": username, "data": data})
    print("Inserted ", inserted_result.inserted_id)
    return inserted_result.inserted_id
