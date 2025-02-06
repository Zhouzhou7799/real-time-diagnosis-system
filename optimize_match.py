import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def get_word_embedding(word_list):
    """ 获取词向量并进行平均处理 """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")
    model.eval()
    encoded = tokenizer(word_list, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**encoded)

    hidden_states = outputs.last_hidden_state
    attention_mask = encoded['attention_mask']

    # 去除[CLS]和[SEP]
    hidden_states = hidden_states[:, 1:-1, :]
    attention_mask = attention_mask[:, 1:-1]

    # 计算平均向量
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    hidden_states_masked = hidden_states * mask_expanded
    sum_embeddings = torch.sum(hidden_states_masked, dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1).unsqueeze(-1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    return mean_embeddings


def calculate_similarity(list1, list2):
    """ 计算两个列表的相似度 """
    embeddings1 = get_word_embedding(list1)
    embeddings2 = get_word_embedding(list2)

    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    similarity_matrix = torch.mm(embeddings1, embeddings2.transpose(0, 1))
    return similarity_matrix.numpy()


def match_zhenghou(extracted_zhenghou, knowledge_base):
    """ 根据患者提取的证候和知识库匹配相应的证型 """
    disease_symptoms = {}

    # 知识库是嵌套列表，遍历第一个列表中的元素
    for knowledge_group in knowledge_base:
        # 遍历每个证候
        for entry in knowledge_group:
            # 获取证候描述
            description = entry['description']

            # 遍历患者提取的证候
            for patient_zhenghou in extracted_zhenghou:
                patient_name = patient_zhenghou['name']
                patient_description = patient_name  # 只用证候名称进行匹配

                # 计算相似度
                similarity_score = calculate_similarity([patient_description], [description])[0][0]

                # 设置相似度阈值，判断是否匹配
                if similarity_score >= 0.8:  # 阈值可以根据需求调整
                    syndrome = entry['Syndrome']
                    sub_syndrome = entry['Sub-Syndrome']

                    # 如果Sub-Syndrome为null，只存储Syndrome，否则存储Syndrome-Sub-Syndrome
                    disease = syndrome
                    if sub_syndrome:
                        disease += f"-{sub_syndrome}"

                    if entry['TCMname'] not in disease_symptoms:
                        disease_symptoms[entry['TCMname']] = set()
                    disease_symptoms[entry['TCMname']].add(disease)

    return disease_symptoms


def load_knowledge_base(file_path):
    """ 从JSON文件加载知识库 """
    with open(file_path, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    return knowledge_base


if __name__ == '__main__':
    # 模拟患者描述数据
    user_description = "小孩肚子疼得厉害，一直发烧到38度"
    extracted_zhenghou = [
        {"name": "肚子疼", "发病部位": "肚子", "性别": None, "程度": "厉害"},
        {"name": "发烧", "发病部位": None, "性别": None, "程度": None}
    ]

    # 假设您的知识库文件路径是 'data_with_similar_zhenghou.json'
    knowledge_base_file = 'modify_data/data_with_similar_zhenghou.json'

    # 从文件加载知识库
    knowledge_base = load_knowledge_base(knowledge_base_file)

    # 进行匹配
    disease_symptoms = match_zhenghou(extracted_zhenghou, knowledge_base)
    print(disease_symptoms)
