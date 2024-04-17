import os
from flask import Flask, request
from services.database_service import DatabaseService
from services.documentation_generation_service import DocumentationGenerationService

app = Flask(__name__)

databaseService = DatabaseService()

documentationGenerationService = DocumentationGenerationService(app.debug)

@app.route("/")
def test():
    return "--> Working!"

@app.route("/generate")
def generate():
  input_method = request.args.get('input_method')
  generated_text = documentationGenerationService.gen_comment(input_method)

  if generated_text:
    return {
       "generated_text": generated_text
    }, 200
  else:
    return "Bad request", 400
  
@app.route("/add_diagnostics", methods=['POST'])
def add_diagnostics():
  username = request.args.get('username')
  data = request.args.get('data')

  try:
    inserted_id = databaseService.insert_record(username, data)
    if inserted_id:
      return {
        "inserted_id": str(inserted_id)
      }, 200
    else:
      return "Bad request", 400
  except Exception as e:
    print(e)
    return "Bad request", 400

if __name__ == "__main__":
    app.run(debug=False)
