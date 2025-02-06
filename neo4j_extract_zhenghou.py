from py2neo import Graph


def query_symptoms_by_syndrome(graph, syndrome_subsyndrome):
    """根据证型-子证型组合查询关联证候"""
    # 分割主证型和子证型（安全分割）
    if '-' not in syndrome_subsyndrome:
        print(f"警告：证型格式错误 '{syndrome_subsyndrome}'，应包含'-'分隔符")
        return set()

    _, sub_syndrome = syndrome_subsyndrome.split('-', 1)

    # 构建精简的Cypher查询
    query = """
    MATCH (sub:中医证型 {name: $sub_syndrome})-[:`leads to`]->(zh:中医症候)
    RETURN zh.name AS symptom
    """

    try:
        return {record['symptom'] for record in graph.run(query, {'sub_syndrome': sub_syndrome})}
    except Exception as e:
        print(f"查询失败: {syndrome_subsyndrome} - {str(e)}")
        return set()




def extract_syndrome_from_graph(matched_syndromes, graph):
    """从知识图谱提取证型对应证候（添加异常处理）"""
    syndrome_dict = {}

    for extracted_name, syndrome_set in matched_syndromes.items():
        for syndrome_subsyndrome in syndrome_set:
            # 验证证型格式
            if '-' not in syndrome_subsyndrome:
                print(f"警告：证型格式异常 - {syndrome_subsyndrome}")
                continue

            # 确保每个证型都有条目
            syndrome_dict.setdefault(syndrome_subsyndrome, set())

            # 查询图谱获取证候
            symptoms = query_symptoms_by_syndrome(graph, syndrome_subsyndrome)
            syndrome_dict[syndrome_subsyndrome].update(symptoms)

    return syndrome_dict


if __name__ == "__main__":
    # Neo4j连接配置（添加超时设置）
    password = "bLdGaCvb93DtXbS9izFcdpCmA2bYK-PXEVS2n29QYV8"
    connect_url = "neo4j+s://c326989f.databases.neo4j.io"


    try:
        # 使用新版推荐连接方式
        graph = Graph(connect_url, auth=("neo4j", password))
        print("✅ Neo4j连接成功")
    except Exception as e:
        print(f"❌ 数据库连接失败: {str(e)}")
        exit(1)

    # 输入数据（保持原有结构）
    matched_syndromes = {
      "胃隐痛": [
        "None-气郁痰瘀",
        "气郁痰瘀"
      ],
      "面色发黄": [
        "None-溃脓期",
        "None-疟疾",
        "内伤咳嗽-肺阴亏虚",
        "发作期-哮喘脱证",
        "实喘-肺气郁痹",
        "尿血-肾气不固",
        "早泄（附）-肾气不固",
        "气厥-虚证",
        "溃脓期",
        "疟疾",
        "血厥-虚证"
      ],
      "食欲不振": [
        "聚证-食滞痰阻"
      ]
    }

    # 从知识图谱提取数据
    try:
        syndrome_dict = extract_syndrome_from_graph(matched_syndromes, graph)
    except Exception as e:
        print(f"数据处理失败: {str(e)}")
        exit(1)

    # 格式化输出结果
    print("\n证型与对应证候关系：")
    for syndrome, symptoms in syndrome_dict.items():
        print(f"{syndrome}: {sorted(list(symptoms))}")  # 按字母顺序排序