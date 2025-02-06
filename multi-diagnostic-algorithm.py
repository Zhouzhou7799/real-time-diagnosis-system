from collections import Counter
import math
from itertools import combinations
import numpy as np

class DiagnosticSystem:
    def __init__(self, disease_symptoms):
        """
        初始化诊断系统
        disease_symptoms: Dict[str, Set[str]] - 疾病名称到症状集合的映射
        """
        self.disease_symptoms = disease_symptoms
        self.all_symptoms = set().union(*disease_symptoms.values())
        
    def calculate_next_n_symptoms(self, current_symptoms, n, method='greedy'):
        """
        计算下一轮应该询问的n个症状
        
        Parameters:
        current_symptoms: Set[str] - 当前已知的症状集合
        n: int - 下一轮需要询问的症状数量
        method: str - 使用的算法方法 ('greedy', 'exhaustive', 'genetic')
        
        Returns:
        Set[str] - 建议询问的n个症状
        """
        candidate_diseases = self._get_candidate_diseases(current_symptoms)
        remaining_symptoms = self.all_symptoms - current_symptoms
        
        if len(remaining_symptoms) < n:
            return remaining_symptoms
            
        if method == 'greedy':
            return self._greedy_selection(current_symptoms, candidate_diseases, remaining_symptoms, n)
        elif method == 'exhaustive':
            return self._exhaustive_search(current_symptoms, candidate_diseases, remaining_symptoms, n)
        elif method == 'genetic':
            return self._genetic_algorithm(current_symptoms, candidate_diseases, remaining_symptoms, n)
        else:
            raise ValueError("Unknown method")

    def _greedy_selection(self, current_symptoms, candidate_diseases, remaining_symptoms, n):
        """
        贪婪算法选择n个症状
        在每一步选择当前最优的症状
        """
        selected_symptoms = set()
        symptoms_list = list(remaining_symptoms)
        
        for _ in range(n):
            if not symptoms_list:
                break
                
            best_score = float('-inf')
            best_symptom = None
            
            # 计算每个症状的边际增益
            for symptom in symptoms_list:
                temp_symptoms = current_symptoms | selected_symptoms | {symptom}
                score = self._calculate_combined_score(temp_symptoms, candidate_diseases)
                
                if score > best_score:
                    best_score = score
                    best_symptom = symptom
            
            if best_symptom:
                selected_symptoms.add(best_symptom)
                symptoms_list.remove(best_symptom)
        
        return selected_symptoms

    def _exhaustive_search(self, current_symptoms, candidate_diseases, remaining_symptoms, n):
        """
        穷举搜索最优的n个症状组合
        警告：当n或症状数量较大时计算量会急剧增加
        """
        best_score = float('-inf')
        best_combination = None
        
        for combo in combinations(remaining_symptoms, n):
            temp_symptoms = current_symptoms | set(combo)
            score = self._calculate_combined_score(temp_symptoms, candidate_diseases)
            
            if score > best_score:
                best_score = score
                best_combination = combo
        
        return set(best_combination) if best_combination else set()

    def _genetic_algorithm(self, current_symptoms, candidate_diseases, remaining_symptoms, n):
        """
        使用遗传算法选择n个症状
        适用于较大规模的症状选择问题
        """
        population_size = 50
        generations = 30
        mutation_rate = 0.1
        
        # 初始化种群
        remaining_list = list(remaining_symptoms)
        population = []
        for _ in range(population_size):
            if len(remaining_list) >= n:
                individual = set(np.random.choice(remaining_list, n, replace=False))
                population.append(individual)
        
        # 进化过程
        for _ in range(generations):
            # 计算适应度
            fitness_scores = []
            for individual in population:
                temp_symptoms = current_symptoms | individual
                score = self._calculate_combined_score(temp_symptoms, candidate_diseases)
                fitness_scores.append(score)
            
            # 选择
            new_population = []
            for _ in range(population_size // 2):
                # 锦标赛选择
                parent1 = self._tournament_select(population, fitness_scores)
                parent2 = self._tournament_select(population, fitness_scores)
                
                # 交叉
                child1, child2 = self._crossover(parent1, parent2, n)
                
                # 变异
                child1 = self._mutate(child1, remaining_symptoms, mutation_rate, n)
                child2 = self._mutate(child2, remaining_symptoms, mutation_rate, n)
                
                new_population.extend([child1, child2])
            
            population = new_population
        
        # 返回最优解
        best_individual = max(population, 
                            key=lambda x: self._calculate_combined_score(current_symptoms | x, candidate_diseases))
        return best_individual

    def _tournament_select(self, population, fitness_scores, tournament_size=3):
        """遗传算法的锦标赛选择"""
        indices = np.random.choice(len(population), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitness)]
        return population[winner_idx]

    def _crossover(self, parent1, parent2, n):
        """遗传算法的交叉操作"""
        # 确保保持每个子代的大小为n
        combined = list(parent1 | parent2)
        if len(combined) < n:
            return parent1, parent2
            
        np.random.shuffle(combined)
        split = len(combined) // 2
        child1 = set(combined[:split])
        child2 = set(combined[split:])
        
        # 调整大小
        while len(child1) < n:
            remaining = parent1 - child1
            if remaining:
                child1.add(np.random.choice(list(remaining)))
        while len(child2) < n:
            remaining = parent2 - child2
            if remaining:
                child2.add(np.random.choice(list(remaining)))
                
        return child1, child2

    def _mutate(self, individual, remaining_symptoms, mutation_rate, n):
        """遗传算法的变异操作"""
        if np.random.random() < mutation_rate:
            available_symptoms = remaining_symptoms - individual
            if available_symptoms:
                # 随机替换一个症状
                individual.remove(np.random.choice(list(individual)))
                individual.add(np.random.choice(list(available_symptoms)))
        return individual

    def _calculate_combined_score(self, symptoms, candidate_diseases):
        """
        计算症状组合的综合得分
        考虑信息增益和症状间的互补性
        """
        # 信息增益分数
        info_gain = self._calculate_total_information_gain(symptoms, candidate_diseases)
        
        # 症状覆盖度分数
        coverage_score = self._calculate_coverage_score(symptoms, candidate_diseases)
        
        # 症状独立性分数
        independence_score = self._calculate_independence_score(symptoms)
        
        # 综合得分（可以调整权重）
        return 0.4 * info_gain + 0.4 * coverage_score + 0.2 * independence_score

    def _calculate_total_information_gain(self, symptoms, candidate_diseases):
        """计算症状组合的总信息增益"""
        initial_entropy = math.log2(len(candidate_diseases))
        
        # 计算使用这些症状后的条件熵
        subgroups = self._split_by_symptoms(symptoms, candidate_diseases)
        conditional_entropy = 0
        total = len(candidate_diseases)
        
        for subgroup in subgroups:
            prob = len(subgroup) / total
            if prob > 0:
                conditional_entropy -= prob * math.log2(prob)
        
        return initial_entropy - conditional_entropy

    def _calculate_coverage_score(self, symptoms, candidate_diseases):
        """计算症状组合对疾病的覆盖程度"""
        total_coverage = 0
        for disease, disease_symptoms in candidate_diseases.items():
            coverage = len(symptoms & disease_symptoms) / len(disease_symptoms)
            total_coverage += coverage
        return total_coverage / len(candidate_diseases)

    def _calculate_independence_score(self, symptoms):
        """计算症状间的独立性分数"""
        if len(symptoms) <= 1:
            return 1.0
            
        # 计算症状间的互信息
        mutual_info = 0
        for s1, s2 in combinations(symptoms, 2):
            mutual_info += self._calculate_mutual_information(s1, s2)
            
        # 转换为独立性分数（互信息越低，独立性越高）
        return 1.0 / (1.0 + mutual_info)

    def _calculate_mutual_information(self, symptom1, symptom2):
        """计算两个症状之间的互信息"""
        count_both = 0
        count_s1 = 0
        count_s2 = 0
        total = len(self.disease_symptoms)
        
        for symptoms in self.disease_symptoms.values():
            has_s1 = symptom1 in symptoms
            has_s2 = symptom2 in symptoms
            if has_s1 and has_s2:
                count_both += 1
            if has_s1:
                count_s1 += 1
            if has_s2:
                count_s2 += 1
        
        if count_both == 0:
            return 0
            
        p_both = count_both / total
        p_s1 = count_s1 / total
        p_s2 = count_s2 / total
        
        if p_both == 0 or p_s1 == 0 or p_s2 == 0:
            return 0
            
        return p_both * math.log2(p_both / (p_s1 * p_s2))

    def _split_by_symptoms(self, symptoms, candidate_diseases):
        """根据症状组合将疾病分组"""
        groups = {}
        for disease, disease_symptoms in candidate_diseases.items():
            # 创建症状特征向量
            key = tuple(symptom in disease_symptoms for symptom in symptoms)
            if key not in groups:
                groups[key] = []
            groups[key].append(disease)
        return list(groups.values())

# 使用示例
def example_usage():
    # 示例疾病-症状数据
    disease_symptoms = {
        "感冒": {"发烧", "咳嗽", "流鼻涕", "头痛", "喉咙痛"},
        "流感": {"高烧", "咳嗽", "肌肉酸痛", "疲劳", "头痛", "恶心"},
        "新冠": {"发烧", "干咳", "疲劳", "嗅觉丧失", "味觉丧失", "呼吸困难"},
        "肺炎": {"高烧", "咳痰", "胸痛", "呼吸困难", "疲劳", "食欲不振"},
        "支气管炎": {"咳嗽", "咳痰", "胸闷", "呼吸困难", "低烧", "疲劳"}
    }
    
    system = DiagnosticSystem(disease_symptoms)
    
    # 第一轮已知症状
    current_symptoms = {"发烧", "咳嗽"}
    
    # 使用不同方法选择3个症状
    print("使用贪婪算法选择症状:")
    next_symptoms = system.calculate_next_n_symptoms(current_symptoms, 3, method='greedy')
    print(next_symptoms)
    
    print("\n使用遗传算法选择症状:")
    next_symptoms = system.calculate_next_n_symptoms(current_symptoms, 3, method='genetic')
    print(next_symptoms)
    
    # 当症状数量较小时，可以使用穷举搜索
    print("\n使用穷举搜索选择症状:")
    next_symptoms = system.calculate_next_n_symptoms(current_symptoms, 2, method='exhaustive')
    print(next_symptoms)
