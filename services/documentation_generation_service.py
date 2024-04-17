import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaConfig, RobertaModel
from cxg.model import Seq2Seq
from cxg.run import convert_examples_to_features, Example

class DocumentationGenerationService():

  def __init__(self, isdebug):
    self.source_length = 256
    self.target_length = 256
    self.beam_size = 10
    self.pretrained_model = "microsoft/codebert-base"
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
    self.config = RobertaConfig.from_pretrained(self.pretrained_model)
    self.encoder = RobertaModel.from_pretrained(self.pretrained_model, config = self.config)
    self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.config.hidden_size, nhead=self.config.num_attention_heads)
    self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
    self.model = Seq2Seq(encoder = self.encoder,decoder = self.decoder,config=self.config,
                    beam_size=self.beam_size,max_length=self.target_length,
                    sos_id=self.tokenizer.cls_token_id,eos_id=self.tokenizer.sep_token_id)

    if not isdebug:
      checkpoint_url = os.environ.get("MODEL_URL")
      self.model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location=self.device))
    else:
      checkpoint_path = "pytorch_model.bin"
      checkpoint = torch.load(checkpoint_path, map_location=self.device)
      self.model.load_state_dict(checkpoint)
    self.model.to(self.device)
    print('Documentation Generation Service Initialized')

  def _prep_input(self, method):
    examples = [
      Example(0, source = method, target = "")
    ]
    args = Args(self.source_length, self.target_length)
    eval_features = convert_examples_to_features(
      examples, 
      self.tokenizer, 
      args,
      stage="test"
    )
    source_ids = torch.tensor(
      eval_features[0].source_ids, 
      dtype = torch.int64
    ).unsqueeze(0).to(self.device)
    source_mask = torch.tensor(
      eval_features[0].source_mask, 
      dtype = torch.int64
    ).unsqueeze(0).to(self.device)
    
    return source_ids, source_mask

  def gen_comment(self, method):
    source_ids, source_mask = self._prep_input(method)
    with torch.no_grad():
      pred = self.model(source_ids = source_ids, source_mask = source_mask)[0]
      t = pred[0].cpu().numpy()
      t = list(t)
      if 0 in t:
        t = t[:t.index(0)]
      text = self.tokenizer.decode(t,clean_up_tokenization_spaces=False)
      return text

class Args:
  def __init__(self, source_length, target_length):
    self.max_source_length = source_length
    self.max_target_length = target_length