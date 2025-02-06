import json


def load_knowledge_base(file_path):
    """ 加载知识库 JSON 文件 """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [item for sublist in raw_data for item in sublist]  # 平铺为一层


def extract_syndrome_from_knowledge_base(matched_syndromes, knowledge_base):
    """ 从知识库中提取每个证型对应的证候名称 """
    syndrome_dict = {}

    # 遍历每个匹配的证型
    for extracted_name, syndrome_set in matched_syndromes.items():
        for syndrome_subsyndrome in syndrome_set:
            if syndrome_subsyndrome not in syndrome_dict:
                syndrome_dict[syndrome_subsyndrome] = set()

            # 遍历知识库，查找对应的证型（Syndrome-Sub-Syndrome）
            for entry in knowledge_base:
                syndrome = entry['Syndrome']
                sub_syndrome = entry.get('Sub-Syndrome', None)
                name = entry['name']

                # 组合证型和子证型
                if sub_syndrome:
                    full_syndrome = f"{syndrome}-{sub_syndrome}"
                else:
                    full_syndrome = syndrome

                # 如果证型（Syndrome-Sub-Syndrome）匹配，则将证候名称（name）添加到字典中
                if full_syndrome == syndrome_subsyndrome:
                    syndrome_dict[syndrome_subsyndrome].add(name)

    return syndrome_dict


# 主程序逻辑
if __name__ == "__main__":
    # 上一步的匹配结果（提取的症候和对应的证型）
    matched_syndromes = {
        "肚子疼": {'寒湿痢-肾阳虚衰'},
        "发烧": {'心营热盛-肾阳虚', '肝经热盛-肾阳虚', '阳黄-疫毒炽盛（急黄）', '虚体感冒-阳虚感冒'}
    }

    # 知识库 JSON 文件路径
    knowledge_base_path = "modify_data/data_with_similar_zhenghou.json"

    # 加载知识库数据
    knowledge_base = load_knowledge_base(knowledge_base_path)

    # 从知识库中提取每个证型对应的证候
    syndrome_dict = extract_syndrome_from_knowledge_base(matched_syndromes, knowledge_base)

    # 打印结果
    print("证型与对应证候关系：")
    for syndrome, symptoms in syndrome_dict.items():
        print(f"{syndrome}: {list(symptoms)}")
