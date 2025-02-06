import json
from Levenshtein import ratio
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def get_word_embedding(word_list):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")
    model.eval()
    encoded = tokenizer(word_list, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded)
    hidden_states = outputs.last_hidden_state
    attention_mask = encoded['attention_mask']
    hidden_states = hidden_states[:, 1:-1, :]
    attention_mask = attention_mask[:, 1:-1]

    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    hidden_states_masked = hidden_states * mask_expanded
    sum_embeddings = torch.sum(hidden_states_masked, dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1).unsqueeze(-1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    return mean_embeddings


def calculate_similarity_matrix(list1, list2):
    embeddings1 = get_word_embedding(list1)
    embeddings2 = get_word_embedding(list2)
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)
    similarity_matrix = torch.mm(embeddings1, embeddings2.transpose(0, 1))

    print("Similarity Matrix:", similarity_matrix.numpy())  # Debugging the similarity matrix
    return similarity_matrix.numpy()


def match_zhenghou(user_zhenghou, knowledge_base, threshold=0.7):
    disease_symptoms = {}

    # 遍历 knowledge_base 中的每个条目
    for symptom in user_zhenghou:
        matched_syndromes = set()
        for entry_list in knowledge_base:  # knowledge_base 是一个包含列表的结构
            for entry in entry_list:  # 遍历每个条目
                if isinstance(entry, dict):  # 确保 entry 是字典类型
                    description = entry.get('description', '')
                    if description:
                        similarity_matrix = calculate_similarity_matrix([symptom['name']], [description])
                        if similarity_matrix[0][0] >= threshold:
                            syndrome = entry.get('Syndrome', '')
                            sub_syndrome = entry.get('Sub-Syndrome', None)
                            if sub_syndrome:
                                matched_syndromes.add(f"{syndrome}-{sub_syndrome}")
                            else:
                                matched_syndromes.add(syndrome)
        disease_symptoms[symptom['name']] = matched_syndromes
    return disease_symptoms


def load_knowledge_base(file_path):
    """
    加载中医证候知识库的 JSON 文件
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    return knowledge_base


# 示例数据
user_description = "医生您好，我最近受凉后容易发病，感觉神疲乏力，并且尿频不畅，请问您能帮我判断一下我的证型吗？我想知道应该采用什么样的治疗方法，以及推荐哪些方药？谢谢"
extracted_zhenghou = [
    {
      "name": "神疲乏力",
      "发病部位": None,
      "性别": None,
      "程度": None
    },
    {
      "name": "尿频不畅",
      "发病部位": "泌尿系统",
      "性别": None,
      "程度": None
    }
  ]


# 假设您有一个 JSON 文件，路径为 'zhenghou_knowledge_base.json'
knowledge_base_file = 'modify_data/data_with_similar_zhenghou.json'
knowledge_base = load_knowledge_base(knowledge_base_file)

# 调用函数进行匹配
disease_symptoms = match_zhenghou(extracted_zhenghou, knowledge_base)
print(disease_symptoms)
