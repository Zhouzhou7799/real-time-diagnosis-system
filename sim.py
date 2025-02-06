from Levenshtein import ratio
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def get_word_embedding(word_list):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")
    model.eval()
    # 批量编码
    encoded = tokenizer(word_list, padding=True, truncation=True, return_tensors='pt')
    # 获取BERT输出
    with torch.no_grad():
        outputs = model(**encoded)
    # 获取CLS向量 (取第一个位置的向量)
    # cls_embeddings = outputs.last_hidden_state[:, 0, :]    
    # return cls_embeddings

    hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
    attention_mask = encoded['attention_mask'] 
    # 去除每个序列的[CLS]和[SEP]标记
    hidden_states = hidden_states[:, 1:-1, :]  # 切片去除首尾标记
    attention_mask = attention_mask[:, 1:-1]   # 相应地调整attention mask
    
    # 计算平均值（考虑attention mask）
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    hidden_states_masked = hidden_states * mask_expanded
    sum_embeddings = torch.sum(hidden_states_masked, dim=1)  # 在序列长度维度求和
    sum_mask = torch.clamp(attention_mask.sum(dim=1).unsqueeze(-1), min=1e-9)  # 防止除零
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings

def calculate_similarity_matrix(list1, list2):
    """
    计算两个词语列表之间的相似度矩阵
        list1: 第一个词语列表
        list2: 第二个词语列表
    Returns:
        相似度矩阵，维度为 len(list1) x len(list2)
    """
    # 获取两个列表的CLS向量
    embeddings1 = get_word_embedding(list1)
    embeddings2 = get_word_embedding(list2)
    # 归一化向量
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)
    # 计算余弦相似度矩阵
    similarity_matrix = torch.mm(embeddings1, embeddings2.transpose(0, 1))
    return similarity_matrix.numpy()


def dsc_bert(prediction, gold, threshold):
    """
    prediction: list, 模型预测的询问实体和概念
    gold: list, 医生真实的询问实体和概念
    """
    similarity_matrix = calculate_similarity_matrix(prediction, gold)
    print(similarity_matrix)
    hit_p = 0
    for i, p in enumerate(prediction):
        for j, g in enumerate(gold):
            if similarity_matrix[i, j]>=threshold:
                print(p, g)
                hit_p += 1
    return (2*hit_p)/(len(prediction)+len(gold))

def dsc_levenshtein(prediction, gold, threshold):
    """
    prediction: list, 模型预测的询问实体和概念
    gold: list, 医生真实的询问实体和概念
    """
    hit_p = 0
    for p in prediction:
        for g in gold:
            if ratio(p, g) >=  threshold:
                hit_p += 1
                print(p)
    return (2*hit_p)/(len(prediction)+len(gold))

def jaccard_levenshtein(prediction, gold, threshold):
    """
    prediction: list, 模型预测的询问实体和概念
    gold: list, 医生真实的询问实体和概念
    """
    hit_p = 0
    for p in prediction:
        for g in gold:
            if ratio(p, g) >=  threshold:
                hit_p += 1
    return hit_p/(len(prediction)-hit_p+len(gold))

def jaccard_bert(prediction, gold, threshold):
    """
    prediction: list, 模型预测的询问实体和概念
    gold: list, 医生真实的询问实体和概念
    """
    similarity_matrix = calculate_similarity_matrix(prediction, gold)
    hit_p = 0
    for i, p in enumerate(prediction):
        for j, g in enumerate(gold):
            if similarity_matrix[i, j]>=threshold:
                print(p, g)
                hit_p += 1
    return hit_p/(len(prediction)-hit_p+len(gold))


if __name__ == '__main__':
    prediction = ["咳嗽", "感冒", "疼痛", "伴随症状", "严重程度", "发烧"]
    gold = ["疼", "伴随症状"]
    print(dsc_bert(prediction, gold, 0.8))
    print(jaccard_bert(prediction, gold, 0.8))