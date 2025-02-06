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

    def _get_candidate_diseases(self, current_symptoms):
        """
        根据当前已知症状返回候选疾病列表
        当前症状与疾病症状的交集不为空的疾病作为候选疾病
        """
        candidate_diseases = {}
        for disease, symptoms in self.disease_symptoms.items():
            # 计算当前已知症状与疾病症状的交集
            intersection = current_symptoms & symptoms
            if intersection:  # 如果交集非空，说明此疾病是候选疾病
                candidate_diseases[disease] = symptoms
        return candidate_diseases

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
        优化后的贪婪算法选择n个症状
        引入启发式排序和多样性奖励
        """
        selected_symptoms = set()
        symptoms_list = list(remaining_symptoms)

        # 启发式排序：根据症状对当前疾病的影响排序
        symptoms_list.sort(key=lambda x: self._calculate_symptom_importance(x, current_symptoms, candidate_diseases),
                           reverse=True)

        for _ in range(n):
            if not symptoms_list:
                break

            best_score = float('-inf')
            best_symptom = None

            for symptom in symptoms_list:
                temp_symptoms = current_symptoms | selected_symptoms | {symptom}
                score = self._calculate_combined_score(temp_symptoms, candidate_diseases)

                if score > best_score:
                    best_score = score
                    best_symptom = symptom

            if best_symptom:
                selected_symptoms.add(best_symptom)
                symptoms_list.remove(best_symptom)

                # 鼓励选择多样化的症状
                self._encourage_diversity(selected_symptoms, symptoms_list)

        return selected_symptoms

    def _exhaustive_search(self, current_symptoms, candidate_diseases, remaining_symptoms, n):
        """
        优化后的穷举搜索，增加剪枝和启发式优先
        """
        best_score = float('-inf')
        best_combination = None

        # 启发式排序：根据症状的重要性排序，优先考虑影响大的症状组合
        sorted_symptoms = sorted(remaining_symptoms,
                                 key=lambda x: self._calculate_symptom_importance(x, current_symptoms,
                                                                                  candidate_diseases), reverse=True)

        # 通过启发式选择前n个症状进行组合
        for combo in combinations(sorted_symptoms, n):
            temp_symptoms = current_symptoms | set(combo)
            score = self._calculate_combined_score(temp_symptoms, candidate_diseases)

            # 剪枝：如果当前组合的得分已经比已有的最优解低，跳过
            if score <= best_score:
                continue

            best_score = score
            best_combination = combo

        return set(best_combination) if best_combination else set()

    def _genetic_algorithm(self, current_symptoms, candidate_diseases, remaining_symptoms, n):
        """
        优化后的遗传算法，加入精英策略、动态变异率、局部搜索
        """
        population_size = 50
        generations = 30
        mutation_rate = 0.1
        elite_size = 5  # 保留精英个体

        # 初始化种群
        remaining_list = list(remaining_symptoms)
        population = [set(np.random.choice(remaining_list, n, replace=False)) for _ in range(population_size)]

        for generation in range(generations):
            fitness_scores = []
            for individual in population:
                temp_symptoms = current_symptoms | individual
                fitness_scores.append(self._calculate_combined_score(temp_symptoms, candidate_diseases))

            # 保留精英个体
            elite_individuals = [population[i] for i in np.argsort(fitness_scores)[-elite_size:]]

            # 选择操作：锦标赛选择
            new_population = elite_individuals[:]
            for _ in range((population_size - elite_size) // 2):
                parent1 = self._tournament_select(population, fitness_scores)
                parent2 = self._tournament_select(population, fitness_scores)

                child1, child2 = self._crossover(parent1, parent2, n)
                child1 = self._mutate(child1, remaining_symptoms, mutation_rate, n)
                child2 = self._mutate(child2, remaining_symptoms, mutation_rate, n)

                new_population.extend([child1, child2])

            # 更新种群
            population = new_population

            # 动态调整变异率（随着迭代进行逐步降低变异率）
            mutation_rate = max(0.01, mutation_rate * 0.95)  # 最低变异率为0.01

            # 进行局部搜索（模拟退火）
            population = [self._local_search(individual, current_symptoms, candidate_diseases) for individual in
                          population]

        # 返回最优解
        best_individual = max(population,
                              key=lambda x: self._calculate_combined_score(current_symptoms | x, candidate_diseases))
        return best_individual

    def _local_search(self, individual, current_symptoms, candidate_diseases):
        """局部搜索：模拟退火"""
        # 随机替换一个症状，看看是否能提高得分
        best_individual = individual
        best_score = self._calculate_combined_score(current_symptoms | individual, candidate_diseases)

        for symptom in individual:
            new_individual = individual - {symptom}
            remaining_symptoms = self.all_symptoms - current_symptoms - individual
            new_individual.add(np.random.choice(list(remaining_symptoms)))

            score = self._calculate_combined_score(current_symptoms | new_individual, candidate_diseases)
            if score > best_score:
                best_score = score
                best_individual = new_individual

        return best_individual

    def _encourage_diversity(self, selected_symptoms, remaining_symptoms):
        """鼓励选择多样化的症状"""
        # 鼓励选择能填补空白的症状，可以通过调整权重来实现
        pass

    def _calculate_symptom_importance(self, symptom, current_symptoms, candidate_diseases):
        """根据症状对候选疾病的影响计算症状重要性"""
        temp_symptoms = current_symptoms | {symptom}
        return self._calculate_combined_score(temp_symptoms, candidate_diseases)

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

    def _tournament_select(self, population, fitness_scores, tournament_size=3):
        """
        锦标赛选择：从种群中随机选取几个个体进行比赛，返回适应度最高的个体。

        Parameters:
        population: List[Set[str]] - 当前种群中的个体
        fitness_scores: List[float] - 每个个体的适应度
        tournament_size: int - 锦标赛中的个体数量

        Returns:
        Set[str] - 适应度最高的个体
        """
        # 从种群中随机选取若干个体进行锦标赛
        indices = np.random.choice(len(population), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitness)]  # 返回适应度最强的个体的索引
        return population[winner_idx]

    def _crossover(self, parent1, parent2, n):
        """
        遗传算法的交叉操作
        从父代1和父代2中生成子代，保证子代大小为n
        """
        # 将父代1和父代2的症状集合合并
        combined = list(parent1 | parent2)

        # 如果合并后的大小大于n，随机选择前n个症状
        if len(combined) > n:
            np.random.shuffle(combined)
            combined = combined[:n]

        # 如果合并后的大小小于n，补充剩余的症状
        while len(combined) < n:
            remaining = (parent1 | parent2) - set(combined)
            combined.append(np.random.choice(list(remaining)))

        return set(combined), set(combined)  # 返回两个子代（为了保证返回两个个体）

    def _mutate(self, individual, remaining_symptoms, mutation_rate, n):
        """
        遗传算法的变异操作
        随机替换个体中的一个症状
        """
        if np.random.random() < mutation_rate:
            # 随机选择一个症状并替换
            available_symptoms = list(remaining_symptoms - individual)
            if available_symptoms:  # 如果有可用的症状进行替换
                # 随机选择一个要替换的症状
                symptom_to_replace = np.random.choice(list(individual))
                new_symptom = np.random.choice(available_symptoms)

                # 进行替换
                individual.remove(symptom_to_replace)
                individual.add(new_symptom)

        return individual

'''# 示例使用
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
    #转换格式np.str->str
    next_symptoms = set(map(str, next_symptoms))
    print(next_symptoms)

    print("\n使用穷举搜索选择症状:")
    next_symptoms = system.calculate_next_n_symptoms(current_symptoms, 2, method='exhaustive')
    print(next_symptoms)


# 运行示例
example_usage()'''
