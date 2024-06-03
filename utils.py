import json
import torch
import numpy as np
import torch.nn.functional as F

def get_config_dict(file_path:str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_and_fix_state_dict(model_pth, prefix_to_remove="0.auto_model."):
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

def modify_state_dict_for_custom_model(state_dict, key_pairs=None):
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

def extract_values(input_ids, one_hot_grad):
    # one_hot_grad の第1次元の長さが input_ids の長さと一致することを確認
    assert one_hot_grad.shape[0] == input_ids.shape[0], "The length of the first dimension of one_hot_grad must match the length of input_ids"

    # 結果を格納するための空のテンソルを作成
    num_tokens = input_ids.numel()
    result = torch.zeros(num_tokens)

    # input_ids の各要素に対応する one_hot_grad の値を抽出して結果に加算
    for i, idx in enumerate(input_ids):
        result[i] = one_hot_grad[i, idx]

    return result


def top_l_percent_threshold(input_grad, l):
    # ソートされたデータを取得（降順）
    sorted_grad = np.sort(input_grad)[::-1]
    
    # トップl％のインデックスを計算
    index = int(np.ceil(len(sorted_grad) * l / 100)) - 1
    
    # インデックスに対応する閾値を取得
    threshold = sorted_grad[index]
    
    return threshold

def normalize_array(array):
    """
    配列を0-1に正規化する関数

    Parameters:
    array (numpy.ndarray): 正規化する配列

    Returns:
    numpy.ndarray: 正規化された配列
    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def comparison_result_html(sorce_text, target_text, grad_list, threshold, output_file_name, title="Results of Sentence Comparison Using Gradients (Identification of Input Tokens with Differences)"):

    # テキストを単語に分割
    words = target_text.split()

    # 各単語とその重要度のリストの長さをチェック
    if len(words) != len(grad_list):
        raise ValueError("The number of words in the text does not match the length of the grad_list.")

    # HTML生成
    html_content = f"<html>\n<head>\n<title>{title}</title>\n</head>\n<body>\n{title}\n<p>\n"

    html_content += f"Source : {sorce_text}<br>Target  : "

    for word, score in zip(words, grad_list):
        if score >= threshold:
            # 赤色の度合いを計算 (しきい値を超えた重要度に基づく)
            intensity = min(int(((score - threshold) / (1.0 - threshold)) * 255), 255)
            color = f"rgb(255, {255 - intensity}, {255 - intensity})"  # 赤色に近づける
            html_content += f'<span style="background-color: {color};">{word}</span> '
        else:
            html_content += f'{word} '

    html_content += "\n</p>\n</body>\n</html>"

    # HTMLファイルに書き込み
    with open(output_file_name, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    print(f"生成されたHTMLファイル: {output_file_name}")


def cosine_similarity_loss(output, target):
    cosine_similarity = F.cosine_similarity(output, target)
    loss = 1 - cosine_similarity.mean()  # 類似度を最大化するために 1 から引く
    return loss