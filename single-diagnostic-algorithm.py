from collections import Counter
import math

class DiagnosticSystem:
    def __init__(self, disease_symptoms):
        """
        初始化诊断系统
        disease_symptoms: Dict[str, Set[str]] - 疾病名称到症状集合的映射
        """
        self.disease_symptoms = disease_symptoms
        self.all_symptoms = set().union(*disease_symptoms.values())
        
    def calculate_next_symptom(self, current_symptoms):
        """
        基于当前症状计算下一个最优询问症状
        current_symptoms: Set[str] - 当前已知的症状集合
        returns: str - 下一个应该询问的症状
        """
        # 获取符合当前症状的候选疾病
        candidate_diseases = self._get_candidate_diseases(current_symptoms)
        
        if not candidate_diseases:
            return None
            
        # 计算剩余症状的信息增益
        remaining_symptoms = self.all_symptoms - current_symptoms
        symptom_scores = {}
        
        for symptom in remaining_symptoms:
            score = self._calculate_information_gain(symptom, candidate_diseases)
            symptom_scores[symptom] = score
            
        # 返回信息增益最大的症状
        return max(symptom_scores.items(), key=lambda x: x[1])[0]
        
    def _get_candidate_diseases(self, current_symptoms):
        """
        获取包含所有当前症状的疾病
        """
        candidates = {}
        for disease, symptoms in self.disease_symptoms.items():
            if current_symptoms.issubset(symptoms):
                candidates[disease] = symptoms
        return candidates
        
    def _calculate_information_gain(self, symptom, candidate_diseases):
        """
        计算某个症状的信息增益
        使用二分类熵来评估症状的区分能力
        """
        total = len(candidate_diseases)
        if total == 0:
            return 0
            
        # 计算当前的熵
        current_entropy = math.log2(total)
        
        # 统计有该症状和没有该症状的疾病数量
        with_symptom = sum(1 for symptoms in candidate_diseases.values() if symptom in symptoms)
        without_symptom = total - with_symptom
        
        # 计算选择该症状后的条件熵
        conditional_entropy = 0
        if with_symptom > 0:
            conditional_entropy += (with_symptom / total) * math.log2(with_symptom)
        if without_symptom > 0:
            conditional_entropy += (without_symptom / total) * math.log2(without_symptom)
            
        # 返回信息增益
        return current_entropy - conditional_entropy

    def get_disease_match_scores(self, symptoms):
        """
        计算症状与各个疾病的匹配度
        returns: Dict[str, float] - 疾病名称到匹配度的映射
        """
        scores = {}
        for disease, disease_symptoms in self.disease_symptoms.items():
            # 计算交集大小除以用户症状数量
            match_rate = len(symptoms & disease_symptoms) / len(symptoms)
            scores[disease] = match_rate
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

# 使用示例
def example_usage():
    # 示例疾病-症状数据
    disease_symptoms = {
        "感冒": {"发烧", "咳嗽", "流鼻涕", "头痛"},
        "流感": {"高烧", "咳嗽", "肌肉酸痛", "疲劳", "头痛"},
        "新冠": {"发烧", "干咳", "疲劳", "嗅觉丧失", "味觉丧失"},
        "肺炎": {"高烧", "咳痰", "胸痛", "呼吸困难", "疲劳"}
    }
    
    system = DiagnosticSystem(disease_symptoms)
    
    # 第一轮已知症状
    current_symptoms = {"发烧", "咳嗽"}
    
    # 获取下一个最优询问症状
    next_symptom = system.calculate_next_symptom(current_symptoms)
    print(f"建议询问的下一个症状: {next_symptom}")
    
    # 计算当前症状与各疾病的匹配度
    scores = system.get_disease_match_scores(current_symptoms)
    print("\n当前症状与各疾病的匹配度:")
    for disease, score in scores.items():
        print(f"{disease}: {score:.2%}")
