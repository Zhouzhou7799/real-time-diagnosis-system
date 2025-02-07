import json
import Levenshtein
import re


def preprocess_text(text):
    """ 预处理文本，去除空格、标点和转换为小写 """
    # 去掉标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 去掉多余的空格，并将文本转换为小写
    text = text.strip().lower()
    return text


def calculate_similarity_with_edit_distance(list1, list2):
    """ 使用编辑距离计算两个列表的相似度 """
    similarity_scores = []
    for item1 in list1:
        for item2 in list2:
            # 预处理文本
            item1 = preprocess_text(item1)
            item2 = preprocess_text(item2)
            # 计算编辑距离并转换为相似度（距离越小，相似度越高）
            distance = Levenshtein.distance(item1, item2)
            max_len = max(len(item1), len(item2))
            similarity_score = 1 - distance / max_len  # 转换为相似度
            similarity_scores.append(similarity_score)
    return similarity_scores


def load_knowledge_base(file_path):
    """ 加载知识库 JSON 文件 """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [item for sublist in raw_data for item in sublist]  # 平铺为一层


def load_extracted_data(file_path):
    """ 加载提取的证候数据 JSON 文件 """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_zhenghou_with_description_only(extracted_zhenghou, knowledge_base, threshold=0.5):
    """ 根据提取的症候名称与知识库的描述匹配，并返回对应的证型和子证型 """
    disease_symptoms = {}

    for extracted in extracted_zhenghou:
        extracted_name = extracted["name"]
        extracted_name_preprocessed = preprocess_text(extracted_name)  # 预处理提取的证候名称

        # 为提取的症候初始化空字典
        disease_symptoms[extracted_name] = set()

        # 遍历知识库，进行描述匹配
        for kb_item in knowledge_base:
            description = kb_item.get("description", "")
            syndrome = kb_item.get("Syndrome", "")
            sub_syndrome = kb_item.get("Sub-Syndrome", None)

            # 预处理知识库的description
            description_preprocessed = preprocess_text(description)

            # 计算编辑距离
            similarity_score = \
            calculate_similarity_with_edit_distance([extracted_name_preprocessed], [description_preprocessed])[0]

            # 打印每次计算的匹配结果，便于调试
            print(f"患者症候: {extracted_name}, 知识库证候描述: {description}, 相似度: {similarity_score}")

            # 判断是否匹配
            if similarity_score >= threshold:  # 如果相似度大于阈值，则认为匹配
                if sub_syndrome:
                    disease_symptoms[extracted_name].add(f"{syndrome}-{sub_syndrome}")
                else:
                    disease_symptoms[extracted_name].add(syndrome)

    return disease_symptoms


# 主程序逻辑
if __name__ == "__main__":
    # 模拟患者描述数据
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

    # 假设您的知识库文件路径是 'data_with_similar_zhenghou.json'
    knowledge_base_path = "../modify_data/data_with_similar_zhenghou.json"

    # 加载提取的证候数据和知识库数据
    knowledge_base = load_knowledge_base(knowledge_base_path)

    # 执行匹配
    disease_symptoms = match_zhenghou_with_description_only(extracted_zhenghou, knowledge_base)

    # 打印结果
    print("证候与病症对应关系：")
    for zhenghou, tcmnames in disease_symptoms.items():
        print(f"{zhenghou}: {tcmnames}")
