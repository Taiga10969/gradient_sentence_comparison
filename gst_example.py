import json
import torch
from transformers import RobertaConfig, RobertaTokenizerFast
from sentence_transformers.models.Pooling import Pooling

from models.modeling_roberta import RobertaModel
from utils import get_config_dict, load_and_fix_state_dict


# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU') 
else: print(f'Use {torch.cuda.device_count()} GPUs')

# 構成ファイルを読み込む
st_roberta_config_pth = './models/st_roberta_config.json'
st_pooling_config_pth = './models/st_pooling_config.json'

roberta_config = get_config_dict(st_roberta_config_pth)
pooling_config = get_config_dict(st_pooling_config_pth)


# Roberta Modelの定義
config = RobertaConfig(**roberta_config)
model = RobertaModel(config)

## 学習済みのパラメータの読み込み
weight_pth = './model_weights/stsb-roberta-large.pt'
fixed_state_dict = load_and_fix_state_dict(weight_pth, prefix_to_remove="0.auto_model.")
msg = model.load_state_dict(fixed_state_dict, strict=True)
print("model.load_state_dict msg : ", msg)
model = model.to(device)
model = model.eval()

# Tokenizerの定義
model_name = 'sentence-transformers/stsb-roberta-large'
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

# Pooling層の定義
pooling = Pooling(**pooling_config)


# Sentence Embedding Process =========================
text = "Water is composed of two hydrogen atoms and one oxygen atom."

inputs = tokenizer(text.lower(), # .lower()によって全て小文字に変換してから処理をする．元々のプログラムでは確認できないが，結果を見ると.lower()をしている．
                   add_special_tokens=True, 
                   return_tensors='pt',
                   )

#print("inputs : ", inputs)
print("inputs(token) : ", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist()))

inputs = {key: value.to(device) for key, value in inputs.items()}

outputs = model(**inputs)

features = {'token_embeddings': outputs.last_hidden_state,
            'attention_mask': inputs['attention_mask'].to(device)
            }

pooling_output = pooling(features)

print("pooling_output['sentence_embedding'] : ", pooling_output['sentence_embedding'])
