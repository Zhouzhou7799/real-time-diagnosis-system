import py2neo
from py2neo import Node, Relationship
import json

from setuptools.command.alias import alias
from tqdm import tqdm
# 连接数据库
password = "bLdGaCvb93DtXbS9izFcdpCmA2bYK-PXEVS2n29QYV8"
connect_url = "neo4j+s://c326989f.databases.neo4j.io"

# Connect to the graph database
graph = py2neo.Graph(connect_url, auth=("neo4j", password))

#加载organ数据
with open('../data/organ_construct.json', 'r', encoding='utf-8') as f:
    organ_data = json.load(f)

#数据NULL值处理
def sanitize_data_with_null_placeholder(data, null_placeholder="NULL"):
    def sanitize(value):
        if value is None:
            return null_placeholder
        elif isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [sanitize(item) for item in value]
        elif isinstance(value, tuple):
            return tuple(sanitize(item) for item in value)
        elif isinstance(value, set):
            return {sanitize(item) for item in value}
        else:
            return value

    return sanitize(data)

# 创建节点
# 创建中医症候节点
def create_zhenghou_node(data):

    #别名属性
    alias = []

    #处理相似症候
    if len(data['similar_zhenghou']) > 1:
        alias = data['similar_zhenghou']

    node_data = {
        "name": data["name"],
        "description": data["description"],
        "alias": alias
    }
    node = Node("中医症候", **node_data)

    #如果节点已经存在，则新增症候节点到证型的关系
    if graph.nodes.match("中医症候", name=data["name"]):
        exist_node = graph.nodes.match("中医症候", name=data["name"]).first()
        return exist_node
    else:
        graph.create(node)
    return node

# 创建中医症型节点(处理父子症型关系)
def create_zhengxing_node(data):
    node_data = {}
    sub_node_data = {}
    #没有子症型
    if data['Sub-Syndrome'] == 'NULL':
        node_data["name"] = data["Syndrome"]
        node_data['group'] = 'NULL'
        node_data['organ'] = organ_data[data["Syndrome"]]
        node = Node("中医证型", **node_data)
        graph.merge(node, "中医证型", "name")
        # 查找父症型节点
        if graph.nodes.match("中医证型", name=data["Syndrome"]):
            tnode = graph.nodes.match("中医证型", name=data["Syndrome"]).first()
        return tnode
    else:
        #存在子症型
        sub_node_data["name"] = data["Sub-Syndrome"]
        sub_node_data["group"] = data["Syndrome"]
        sub_node_data['organ'] = organ_data[data["Sub-Syndrome"]]

        node_data["name"] = data["Syndrome"]
        node_data['group'] = 'NULL'
        node_data['organ'] = organ_data[data["Syndrome"]]

        sub_node = Node("中医证型", **sub_node_data)
        node = Node("中医证型", **node_data)

        #创建节点
        graph.merge(sub_node, "中医证型", "name")
        graph.merge(node, "中医证型", "name")

        # 查找子症型节点
        if graph.nodes.match("中医证型", name=data["Sub-Syndrome"]):
            tsub_node = graph.nodes.match("中医证型", name=data["Sub-Syndrome"]).first()

        # 查找父症型节点
        if graph.nodes.match("中医证型", name=data["Syndrome"]):
            tnode = graph.nodes.match("中医证型", name=data["Syndrome"]).first()

        relationship = Relationship(tsub_node, "part of", tnode)
        graph.merge(relationship)
        return tsub_node


#创建西医疾病节点
def create_western_disease_node(data):
    node_data = {}
    node_data["name"] = data["TCMname"]
    node = Node("西医疾病", **node_data)
    graph.merge(node, "西医疾病", "name")
    return node

# 创建部位节点
def create_part_node(data, zhenghou_node):
    data = data['structured_json']
    if isinstance(data['Part'], list):
        for part in data['Part']:
            if part != 'NULL':
                node_data = {"name": part}
                node = Node("部位", **node_data)
                graph.merge(node, "部位", "name")
                relationship = Relationship(zhenghou_node, "affected on", node)
                graph.merge(relationship)
    else:
        if data['Part'] != 'NULL':
            node_data = {"name": data["Part"]}
            node = Node("部位", **node_data)
            graph.merge(node, "部位", "name")
            relationship = Relationship(zhenghou_node, "affected on", node)
            graph.merge(relationship)

# 创建程度节点
def create_level_node(data):
    data = data['structured_json']
    if data['Level'] != 'NULL':
        node_data = {"name": data["Level"]}
        node = Node("程度", **node_data)
        graph.merge(node, "程度", "name")
        return node

# 创建发作时间节点
def create_time_node(data):
    data = data['structured_json']
    if data['Time'] != 'NULL':
        node_data = {"name": data["Time"]}
        node = Node("发作时间", **node_data)
        graph.merge(node, "发作时间", "name")
        return node

# 创建性别节点
def create_gender_node(data):
    data = data['structured_json']
    if data['Gender'] == '女性'or data['Gender'] == '男性':
        node_data = {"name": data["Gender"]}
        node = Node("性别", **node_data)
        graph.merge(node, "性别", "name")
        return node

if __name__ == '__main__':
    with open('../data/data_with_similar_zhenghou.json', 'r', encoding='utf-8') as f:
        data = json.load(f)



    for i in data:
        print(type(i))
        for item in tqdm(i):
            print(type(item))
            try:
                item = sanitize_data_with_null_placeholder(item)
                #print(f"Processing item: {item}")

                zhenghou_node = create_zhenghou_node(item)
                zhengxing_node = create_zhengxing_node(item)
                tcm_node = create_western_disease_node(item)

                level_node = create_level_node(item)
                time_node = create_time_node(item)
                gender_node = create_gender_node(item)

                #创建关系

                #创建症型与症候关系
                relationship = Relationship(zhengxing_node, "leads to", zhenghou_node)
                graph.merge(relationship)

                #创建症型和西医疾病关系
                relationship = Relationship(tcm_node, "caused by", zhengxing_node)
                graph.merge(relationship)

                if level_node:
                    relationship = Relationship(zhenghou_node, "has a", level_node)
                    graph.merge(relationship)

                if "structured_json" in item and "Part" in item['structured_json']:
                    create_part_node(item, zhenghou_node)

                if gender_node:
                    relationship = Relationship(zhenghou_node, "related to", gender_node)
                    graph.merge(relationship)

                if time_node:
                    relationship = Relationship(zhenghou_node, "occurs at", time_node)
                    graph.merge(relationship)



            except KeyError as ke:
                print(f"KeyError occurred for item {item}: {ke}")
            except py2neo.DatabaseError as de:
                print(f"DatabaseError occurred: {de}")
            except Exception as e:
                print(f"Unexpected error occurred: {e}")
