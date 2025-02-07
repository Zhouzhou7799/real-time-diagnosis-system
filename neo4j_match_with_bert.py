from neo4j import GraphDatabase
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json
from typing import List, Dict, Any
import logging



# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """BERT 嵌入服务单例"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init_model()
        return cls._instance

    def init_model(self):
        """初始化模型并移至GPU（如果可用）"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing BERT model on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.model = AutoModel.from_pretrained("bert-base-chinese").to(self.device)
        self.model.eval()

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """批量获取文本嵌入"""
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state

            # 动态去除[CLS]和[SEP]
            attention_mask = inputs.attention_mask[:, 1:-1]
            hidden_states = hidden_states[:, 1:-1, :]

            # 加权平均池化
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1).unsqueeze(-1), min=1e-9)

        return (sum_embeddings / sum_mask).cpu()


class Neo4jConnector:
    """优化的Neo4j连接器"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_indexes()

    def _create_indexes(self):
        """创建必要的数据库索引"""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (s:中医症候) ON (s.name)",
            "CREATE INDEX IF NOT EXISTS FOR (z:中医证型) ON (z.name)",
            "CREATE INDEX IF NOT EXISTS FOR (p:部位) ON (p.name)"
        ]
        with self.driver.session() as session:
            for query in index_queries:
                session.run(query)

    def close(self):
        self.driver.close()

    def get_all_syndromes(self) -> Dict[str, dict]:
        """获取所有症候及关联证型"""
        query = """
        MATCH (s:中医症候)
        OPTIONAL MATCH (s)<-[:`leads to`]-(z:中医证型)
        OPTIONAL MATCH (s)-[:`affected on`]->(p:部位)
        WITH s, z, collect(DISTINCT p.name) AS locations
        RETURN s.name AS syndrome_name,
               s.description AS syndrome_desc,
               coalesce(collect(DISTINCT z.name), []) AS related_zhengxing,
               locations
        """
        with self.driver.session() as session:
            result = session.run(query)
            return {
                record["syndrome_name"]: {
                    "description": record["syndrome_desc"],
                    "zhengxing": record["related_zhengxing"],
                    "locations": record["locations"]
                } for record in result
            }


class SyndromeMatcher:
    def __init__(self, connector: Neo4jConnector):
        self.connector = connector
        self.embedder = EmbeddingService()
        self.syndrome_cache = None
        self._preload_syndromes()  # 在构造函数中调用此方法

    def _preload_syndromes(self):
        """预加载症候数据并缓存"""
        if self.syndrome_cache is None:
            print("Loading syndromes from graph database...")
            self.syndrome_cache = self.connector.get_all_syndromes()
            print(f"Loaded {len(self.syndrome_cache)} syndromes.")

    def _batch_similarity(self, symptoms: List[str]) -> Dict[str, List[tuple]]:
        """批量相似度计算"""
        self._preload_syndromes()

        # 准备文本数据
        symptom_texts = symptoms
        syndrome_items = list(self.syndrome_cache.items())
        syndrome_texts = [desc["description"] for _, desc in syndrome_items]

        # 批量生成嵌入
        all_embeddings = self.embedder.get_embeddings(symptom_texts + syndrome_texts)
        symptom_embeds = all_embeddings[:len(symptoms)]
        syndrome_embeds = all_embeddings[len(symptoms):]

        # 计算相似度矩阵
        symptom_embeds = F.normalize(symptom_embeds, p=2, dim=1)
        syndrome_embeds = F.normalize(syndrome_embeds, p=2, dim=1)
        sim_matrix = torch.mm(symptom_embeds, syndrome_embeds.T)

        # 构建结果映射
        matches = {}
        for i, symptom in enumerate(symptoms):
            scores = []
            for j, (syndrome_name, syndrome_data) in enumerate(syndrome_items):
                score = sim_matrix[i][j].item()
                scores.append((syndrome_name, score))
            matches[symptom] = scores
        return matches

    def match_zhenghou(self, user_zhenghou: List[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, list]:
        if self.syndrome_cache is None:
            raise ValueError("Syndrome cache has not been initialized. Call _preload_syndromes first.")

        symptoms = [s["name"] for s in user_zhenghou]
        syndrome_items = list(self.syndrome_cache.items())
        syndrome_texts = [desc["description"] for _, desc in syndrome_items]

        all_embeddings = self.embedder.get_embeddings(symptoms + syndrome_texts)
        symptom_embeds = all_embeddings[:len(symptoms)]
        syndrome_embeds = all_embeddings[len(symptoms):]

        symptom_embeds = F.normalize(symptom_embeds, p=2, dim=1)
        syndrome_embeds = F.normalize(syndrome_embeds, p=2, dim=1)
        sim_matrix = torch.mm(symptom_embeds, syndrome_embeds.T)

        matches = {}
        detailed_matches = {}
        for i, symptom in enumerate(symptoms):
            scores = []
            for j, (syndrome_name, syndrome_data) in enumerate(syndrome_items):
                score = sim_matrix[i][j].item()
                scores.append((syndrome_name, score))
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
            matches[symptom] = sorted_scores

            # 记录详细的匹配信息，包括相似度分数和对应的中医证候名称
            detailed_matches[symptom] = [{"syndrome_name": name, "similarity_score": score} for name, score in
                                         sorted_scores]

        final_results = {symptom: {"matched_zhengxing": [], "similar_syndromes": []} for symptom in symptoms}

        for symptom, scores in matches.items():
            zhengxing_relations = set()
            for syndrome_name, score in scores:
                if score >= threshold:
                    query = """
                    MATCH (s:中医症候 {name:$syndrome_name})<-[:`leads to`]-(subz:中医证型)
                    OPTIONAL MATCH (subz)-[:`part of`]->(z:中医证型)
                    WHERE subz IS NOT NULL
                    RETURN z.name AS zhengxing, subz.name AS subzhengxing
                    UNION ALL
                    MATCH (s:中医症候 {name:$syndrome_name})<-[:`leads to`]-(z:中医证型)
                    WHERE NOT EXISTS((z)-[:`part of`]->())
                    RETURN z.name AS zhengxing, null AS subzhengxing
                    """
                    with self.connector.driver.session() as session:
                        result = session.run(query, {"syndrome_name": syndrome_name})
                        for record in result:
                            zhengxing = record['zhengxing']
                            subzhengxing = record['subzhengxing']
                            if subzhengxing and not subzhengxing.startswith('None'):
                                zhengxing_relations.add(f"{zhengxing}-{subzhengxing}")
                            elif not subzhengxing:
                                zhengxing_relations.add(zhengxing)

            # 去除重复的证型，并且保持“中医证型-子证型”的格式，对于没有子证型的中医证型，保留其原样
            cleaned_zhengxing_relations = set()
            seen_main_types = {}
            for item in zhengxing_relations:
                if '-' in item:
                    main_type, sub_type = item.split('-', 1)
                    if main_type not in seen_main_types or seen_main_types[main_type].startswith(main_type):
                        cleaned_zhengxing_relations.add(item)
                        seen_main_types[main_type] = item
                else:
                    if item not in seen_main_types:
                        cleaned_zhengxing_relations.add(item)
                        seen_main_types[item] = item

            final_results[symptom]["matched_zhengxing"] = sorted(list(cleaned_zhengxing_relations),
                                                                 key=lambda x: (x.split('-')[0] if '-' in x else x,
                                                                                x.split('-')[1] if '-' in x else ''))
            final_results[symptom]["similar_syndromes"] = detailed_matches[symptom]

        return final_results

if __name__ == '__main__':
     # 初始化组件
    connector = Neo4jConnector(
            "neo4j+s://c326989f.databases.neo4j.io",
            "neo4j",
            "bLdGaCvb93DtXbS9izFcdpCmA2bYK-PXEVS2n29QYV8"
    )
    matcher = SyndromeMatcher(connector)

    #加载提取的证候信息
    with open('data/extracted_output.json', 'r', encoding='utf-8') as f:
         extracted_zhenghou_data = json.load(f)

     # 准备用于匹配的证候列表
    patient_zhenghou_list = []
    for item in extracted_zhenghou_data:
         if 'extracted_zhenghou' in item and isinstance(item['extracted_zhenghou'], list):
             for zhenghou in item['extracted_zhenghou']:
                 patient_zhenghou_list.append({
                     "name": zhenghou.get("name"),
                     "发病部位": zhenghou.get("发病部位"),
                     "性别": zhenghou.get("性别"),
                     "程度": zhenghou.get("程度")
                 })

    # 执行匹配
    result = matcher.match_zhenghou(patient_zhenghou_list, threshold=0.8)

    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 关闭连接
    connector.close()