import json
import torch
from torch import nn
import torch.nn.functional as F

from sentence_transformers.models.Pooling import Pooling
from transformers import RobertaConfig, RobertaTokenizerFast

from models.modeling_roberta_replace_embeddings import CUSTOM_RobertaModel

class gradient_sentence_transformer(nn.Module):
    def __init__(self, 
                 base_model_name = 'sentence-transformers/stsb-roberta-large',
                 st_roberta_config_pth = './models/st_roberta_config.json', 
                 st_pooling_config_pth = './models/st_pooling_config.json', 
                 weight_pth = None, 
                 device = None,
                 ):
        super().__init__()
        
        self.device = device
        self.tokenizer = RobertaTokenizerFast.from_pretrained(base_model_name)

        self.roberta_config = self.get_config_from_json(st_roberta_config_pth)
        self.pooling_config = self.get_config_from_json(st_pooling_config_pth)


        config = RobertaConfig(**self.roberta_config)
        self.model = CUSTOM_RobertaModel(config)

        if weight_pth is not None:
            fixed_state_dict = self.load_and_fix_state_dict(weight_pth, prefix_to_remove="0.auto_model.")
            new_state_dict = self.modify_state_dict_for_custom_model(fixed_state_dict)
            msg = self.model.load_state_dict(new_state_dict, strict=True)
            print("model.load_state_dict msg : ", msg)

        self.model = self.model.to(device)
        self.model = self.model.eval()
        self.pooling = Pooling(**self.pooling_config)
    

    def load_and_fix_state_dict(self, model_pth, prefix_to_remove="0.auto_model."):
        # モデルの状態辞書を読み込み
        state_dict = torch.load(model_pth, map_location=torch.device('cpu'))
        
        # 不一致するキーの修正
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix_to_remove):
                new_key = key.replace(prefix_to_remove, "")  # 不一致する部分を修正
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
                
        return new_state_dict
    

    def modify_state_dict_for_custom_model(self, state_dict, key_pairs=None):
        if key_pairs is None:
            key_pairs = {'embeddings.word_embeddings.weight': 'embeddings.embedding_projection.weight'}
        
        new_state_dict = {}
        for key, value in state_dict.items():
            if key in key_pairs:
                new_key = key_pairs[key]
                new_state_dict[new_key] = value.t()  # 転置して形状を [hidden_size, vocab_size] に変更
            else:
                new_state_dict[key] = value
        return new_state_dict
    

    def get_config_from_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    

    def input_preparation(self, text):
        inputs = self.tokenizer(text.lower(), # .lower()によって全て小文字に変換してから処理
                                add_special_tokens=True, 
                                return_tensors='pt',
                                )
        
        one_hot_input_ids = F.one_hot(inputs['input_ids'], num_classes=self.roberta_config['vocab_size']).float()

        return inputs, one_hot_input_ids
    

    def convert_ids_to_tokens(self, inputs):
        return self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
    

    def forward(self, inputs, one_hot_input_ids, pooling=True):
        outputs = self.model(input_ids = inputs['input_ids'].to(self.device),
                             attention_mask = inputs['attention_mask'].to(self.device),
                             one_hot_input_ids = one_hot_input_ids.to(self.device),
                             )
        
        if pooling == True:
            features = {'token_embeddings': outputs.last_hidden_state,
                        'attention_mask': inputs['attention_mask'].to(self.device)
                        }
            outputs = self.pooling(features)
        
        return outputs


    







if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count()==0: print('Use 1 GPU') 
    else: print(f'Use {torch.cuda.device_count()} GPUs')

    embeddings_model = gradient_sentence_transformer(device = device)


    text = 'I am Taiga.'

    inputs, one_hot_input_ids = embeddings_model.input_preparation(text)
    
    outputs = embeddings_model.forward(inputs = inputs, 
                                       pooling = True,
                                       )
