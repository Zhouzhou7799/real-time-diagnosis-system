import pandas as pd
import json

# 读取Excel文件
table_data = pd.read_excel('2_内科病症.xlsx')  # 替换为你的Excel文件路径

# 提取关键列，并确保只有当“证候”列不为空时才提取
table_records = table_data[table_data['证候'].notnull()][['病症名', '证型', '子证型']].to_dict(orient='records')

# 加载JSON数据
with open('stru_environment.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)


def match_and_correct(json_data, table_records):
    # 创建一个映射表，用于快速查找
    table_map = {(record['病症名'], record['证型']): (record['子证型'] if not pd.isnull(record['子证型']) else None) for
                 record in table_records}

    corrected_json_data = []

    for record in json_data:
        tcm_name = record['TCMname']
        syndrome = record['Syndrome']

        # 构建键值对
        key = (tcm_name, syndrome)

        if key in table_map:
            # 如果找到对应的记录，则更新记录
            record['Sub-Syndrome'] = table_map[key]
        else:
            # 对于不能完全匹配的记录，保持原样或设置Sub-Syndrome为None
            record['Sub-Syndrome'] = None

        corrected_json_data.append(record)

    return corrected_json_data


# 调用函数进行匹配和修正
corrected_json_data = match_and_correct(json_data, table_records)


# 保存修正后的JSON数据
def save_to_json_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


save_to_json_file(corrected_json_data, 'corrected_environment.json')

print("修正完成，结果已保存到 corrected_environment.json")