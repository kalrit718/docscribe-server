import os
from flask import Flask, request
from services.database_service import DatabaseService
import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaConfig, RobertaModel
from model import Seq2Seq
from run import convert_examples_to_features, Example

app = Flask(__name__)

databaseService = DatabaseService()

source_length = 256
target_length = 256
beam_size = 10
pretrained_model = "microsoft/codebert-base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
config = RobertaConfig.from_pretrained(pretrained_model)
encoder = RobertaModel.from_pretrained(pretrained_model, config = config)
decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
model = Seq2Seq(encoder = encoder,decoder = decoder,config=config,
                beam_size=beam_size,max_length=target_length,
                sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

checkpoint_url = os.environ.get("MODEL_URL")
model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location=device))
model.to("cpu")

class Args:
    max_source_length = source_length
    max_target_length = target_length

args = Args()

@app.route("/")
def test():
    return "--> Working!"

@app.route("/generate")
def generate():
  input_method = request.args.get('input_method')
  generated_text = gen_comment(input_method)

  if generated_text:
    return {
       "generated_text": generated_text
    }, 200
  else:
    return "Bad request", 400
  
def prep_input(method):
  examples = [
     Example(0, source = method, target = "")
  ]
  eval_features = convert_examples_to_features(
    examples, 
    tokenizer, 
    args, 
    stage="test"
  )
  source_ids = torch.tensor(
    eval_features[0].source_ids, 
    dtype = torch.int64
  ).unsqueeze(0).to("cpu")
  source_mask = torch.tensor(
    eval_features[0].source_mask, 
    dtype = torch.int64
  ).unsqueeze(0).to("cpu")
  
  return source_ids, source_mask

def gen_comment(method):
  source_ids, source_mask = prep_input(method)
  with torch.no_grad():
    pred = model(source_ids = source_ids, source_mask = source_mask)[0]
    t = pred[0].cpu().numpy()
    t = list(t)
    if 0 in t:
      t = t[:t.index(0)]
    text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
    return text
  
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
    app.run(debug=True)