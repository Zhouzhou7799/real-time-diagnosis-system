
import logging
import json
import re
import time
from openai import OpenAI

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url='https://api.key77qiqi.cn/v1',
    api_key='sk-PC6YTiFwgwYQNbNYJJ3Br20k9u4yBv7rMtdwSvTBlRkQlNmP'
)

# 输入和输出文件
input_path = 'data/input.json'
output_path = 'data/extracted_output.json'

# 初始化空文件
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump([], f, ensure_ascii=False, indent=2)

# 清理返回内容的函数
def clean_model_response(response_content):
    """
    清理模型返回的内容，去掉多余的换行符、转义字符以及代码块标记，确保为合法 JSON。
    """
    # 去掉代码块标记 ```json 和 ```
    cleaned_content = re.sub(r"```json\n|\n```", "", response_content).strip()

    # 去掉多余的换行符和转义符，使其成为单行 JSON 字符串
    cleaned_content = cleaned_content.replace("\n", "").replace("\r", "").replace("\\", "").strip()

    return cleaned_content

def process_user_input(data, batch_size=5, delay=1):
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        for item in batch:
            try:
                # 优化提示词
                request_content = (
                    f"请提取以下句子中的证候信息，并严格按照 JSON 格式返回结果，不需要额外的文字说明。"
                    f"每个证候需要包含以下字段："
                    f"- name（证候名称）"
                    f"- 发病部位"
                    f"- 性别"
                    f"- 程度"
                    f"注意："
                    f"1. 只返回 JSON 格式内容，不需要任何附加说明。"
                    f"2. 如果某些字段缺失（如性别或程度未提及），请设置为 `null`。"
                    f"现在提取以下句子的证候信息："
                    f"“{item['description']}”"
                )
                # 调用模型
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": request_content}],
                    model="gpt-4o-mini"
                )
                content = response.choices[0].message.content
                cleaned_content = clean_model_response(content)

                # 验证 JSON 格式
                try:
                    extracted_zhenghou = json.loads(cleaned_content)
                except json.JSONDecodeError:
                    # 尝试从说明性文字中提取 JSON
                    match = re.search(r"\[.*\]", cleaned_content, re.DOTALL)
                    if match:
                        cleaned_content = match.group(0)
                        extracted_zhenghou = json.loads(cleaned_content)
                    else:
                        raise ValueError(f"Non-JSON response: {cleaned_content}")

                # 保存结果
                result = {
                    "user_description": item["description"],
                    "extracted_zhenghou": extracted_zhenghou
                }
                with open(output_path, 'r+', encoding='utf-8') as f:
                    current_data = json.load(f)
                    current_data.append(result)
                    f.seek(0)
                    json.dump(current_data, f, ensure_ascii=False, indent=2)

                print(f"Processed input: {item['description']}")
                print(json.dumps(result, ensure_ascii=False, indent=2))

            except ValueError as e:
                print(f"ValueError: {e}")
                with open(output_path, 'r+', encoding='utf-8') as f:
                    current_data = json.load(f)
                    current_data.append({"user_description": item["description"], "error": str(e)})
                    f.seek(0)
                    json.dump(current_data, f, ensure_ascii=False, indent=2)

            time.sleep(delay)

# 主逻辑
# if __name__ == "__main__":
#     with open(input_path, 'r', encoding='utf-8') as f:
#         user_input_data = json.load(f)
#
#     process_user_input(user_input_data)