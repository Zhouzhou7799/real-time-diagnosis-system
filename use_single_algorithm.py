import json
import Levenshtein
import math
from collections import Counter


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


# 数据处理和初始化
def process_input_data(input_data):
    """
    将输入的字典数据转换为疾病-症状的集合形式
    """
    disease_symptoms = {}

    for syndrome, symptoms in input_data.items():
        # 证型是症候的集合
        for symptom in symptoms:
            if syndrome not in disease_symptoms:
                disease_symptoms[syndrome] = set()
            disease_symptoms[syndrome].add(symptom)

    return disease_symptoms


# 示例输入数据
input_data = {
    'None-气郁痰瘀': ['便溏', '呕血', '咳嗽咳痰', '善太息', '痰白', '痰质稠黏', '痰黄白相兼', '神疲乏力', '胸膈痞闷', '脉细涩', '脘腹刺痛', '脘腹胀满', '脘腹胀痛不适', '脘腹隐痛', '舌质暗隐紫', '苔薄腻', '黑便'],
    'None-溃脓期': ['咳吐大量脓血痰', '咳血', '气喘不能卧', '烦渴喜饮', '胸中烦满而痛', '脉数实', '脉滑数', '舌质红', '苔黄腻', '身热', '面赤'],
    'None-疟疾': ['口渴引饮', '哈欠乏力', '壮热', '头痛', '寒战', '寒栗鼓颔', '热退身凉', '脉象弦', '舌红', '苔薄白', '苔黄腻', '遍身汗出', '面赤'],
    '内伤咳嗽-肺阴亏虚': ['上气咳逆阵作', '乏力', '午后潮热', '口干咽燥', '口干欲饮', '咳吐不爽', '咳嗽反复发作', '咳嗽气粗', '咳声短促', '咳声重浊', '咳时引痛', '咳时面红目赤，引胸胁作痛', '咽干口苦', '喉中血丝', '痰中带血丝', '痰多色白', '痰多黄稠', '痰如絮条', '痰少质黏色白', '痰有热腥味', '痰滞咽喉，咳之难出', '痰质黏', '痰量少', '痰黏厚', '痰黏腻或稠厚成块', '盗汗', '神疲乏力', '纳差', '胸胁胀满', '胸苔薄黄', '苔薄黄腻', '身热', '面赤', '颧红'],
    '发作期-哮喘脱证': ['不恶寒', '发前自觉咽发痒', '发前自觉眼发痒', '发前自觉耳发痒', '发前自觉鼻发痒', '发热', '口不渴', '口唇爪甲青紫', '口干欲饮', '口渴喜饮', '口苦', '呼吸急促', '咳不甚', '咳吐不利', '咳吐不爽', '咳呛阵作', '咳痰不爽', '咳痰无力', '咳痰色白', '咳痰色黄', '咳痰黏腻难出', '咽干口渴', '哮病反复久发', '唇紫', '喉中哮鸣', '喉中哮鸣如鼾', '喉中哮鸣有声', '喉中痰涎壅盛，声如拽锯，或鸣声如吹哨笛', '喉中鸣息有声', '喘咳气逆', '喘急胸满', '喘息鼻扇，张口抬肩', '喷嚏', '四肢厥冷', '坐不得卧', '声低', '大便偏干', '天冷或受寒易发', '头痛', '形寒', '形寒肢冷', '息促', '恶寒', '持续哮喘', '无明显寒热倾向', '无汗', '气短', '气短息促', '气粗息涌', '汗出', '汗出如油', '流涕', '渴喜热饮', '烦热', '烦躁', '烦闷不安', '畏冷', '痰涎清稀', '痰稀薄色白', '痰质黏起沫', '痰黄白相兼', '痰黏浊稠厚', '痰黏色黄', '白色泡沫痰液', '神志昏蒙', '胸膈满闷如塞', '胸膈烦闷', '胸部憋塞', '胸高胁胀', '脉弦滑', '脉弦紧', '脉沉细', '脉浮大无根', '脉浮紧', '脉滑实', '脉滑数', '脉细数', '脉细数不清', '舌尖边红', '舌苔厚浊', '舌苔白滑', '舌苔白腻罩黄', '舌质偏红', '舌质淡', '舌质紫暗', '舌质红', '舌质青暗', '苔腻', '苔黄腻', '身痛', '面色晦滞带青', '面色苍白', '面色青暗', '面赤', '面青', '颧红', '鼻塞'],
    '实喘-肺气郁痹': ['不渴', '发热', '口不渴', '口渴', '口黏', '吐痰稠黏', '呕恶', '呼吸急促', '咳吐不利', '咳而不爽', '咽中如窒', '咽干', '喉中痰鸣不著', '喘咳气涌', '喘咳痰鸣', '喘逆上气', '喜冷饮', '大便', '失眠', '头痛', '小便赤涩', '平素多忧思抑郁', '形寒', '心悸', '心烦易怒', '息粗', '息粗气憋', '恶寒', '无汗', '有汗', '烦闷', '痰多', '痰多质黏', '痰多黏腻', '痰色白', '痰色白清稀', '痰色黄', '突然呼吸短促', '纳呆', '胸中满闷', '胸中烦闷', '胸痛', '胸盈仰息', '胸胀', '胸胁闷痛', '胸部胀痛', '胸部胀闷', '脉弦', '脉浮数', '脉浮紧', '脉滑', '脉滑数', '脉濡', '舌苔薄白而滑', '舌质淡', '舌质红', ' 苔白腻', '苔薄白', '苔薄黄', '苔黄', '苔黄腻', '血痰', '身热', '身热有汗', '身痛', '面红目赤', '面赤', '鼻塞', '鼻扇'],
    '尿血-肾气不固': ['久病尿血', '五心烦热', '体倦乏力', '便溏', '健忘', '口渴', '口疮', '口苦', '咽干', '声低', '多梦', '夜寐不安', '夜尿多', '夜尿清长', '头昏', '头晕', '小便清长', '小便短赤带血', '小便黄', '小便黄赤灼热', '尿血色淡', '尿血量多', '尿血鲜红', '形寒肢冷', '心悸', '心烦', '怔忡', '性欲减退', '无梦而遗', '早泄', '早泄遗精', '时作时止', '气短', '滑精不禁', '潮热', '盗汗', '神疲乏力', '精液清冷', '精神困惫', '耳鸣', '肌衄', '胁痛', '胸闷', '脉弦滑', '脉数', '脉沉弱', '脉沉细', '脉细弱', '脉细数', '腰脊酸痛', '腰膝酸软', '腹胀', '舌淡', '舌淡苔白', '舌红', '舌红少苔', '舌质淡胖而嫩', ' 苔少', '苔白滑', '苔黄腻', '血色淡红', '遗精频作', '阳事易举', '阳痿', '阴囊湿痒', '阴茎易举', '面色㿠白', '面色不华', '面赤', '颧红潮热', '食少', '齿衄'],
    '早泄（附）-肾气不固': ['久病尿血', '五心烦热', '体倦乏力', '便溏', '健忘', '口渴', '口疮', '口苦', '咽干', '声低', '多梦', '夜寐不安', '夜尿多', '夜尿清长', '头昏', '头晕', '小便清长', '小便短赤带血', '小便黄浊', '小便黄赤灼热', '尿血色淡', '尿血量多', '尿血鲜红', '形寒肢冷', '心悸', '心烦', '怔忡', '性欲减退', '无梦而遗', '早泄', '早泄遗精', '时作时止', '气短', '滑精不禁', '潮热', '盗汗', '神疲乏力', '精液清冷', '精神困惫', '耳鸣', '肌衄', '胁痛', '胸闷', '脉弦滑', '脉数', '脉沉弱', '脉沉细', '脉细弱', '脉细数', '腰脊酸痛', '腰膝酸软', '腹胀', '舌淡', '舌淡苔白', '舌红', '舌红少苔', '舌质淡胖而嫩', '苔少', '苔白滑', '苔黄腻', '血色淡红', '遗精频作', '阳事易举', '阳痿', '阴囊湿痒', '阴茎易举', '面色㿠白', '面色不华', '面赤', '颧红潮热', '食少', '齿衄'],
    '气厥-虚证': ['不省人事', '口唇无华', '口噤握拳', '呼吸微弱', '呼吸气粗', '唇紫', '四肢厥冷', '四肢震颤', '失血过多', '恐惧', '情志异常', '情绪紧张', '汗出肢冷', '沉弦', '牙关紧闭', '疼痛', '目陷口张', '眩晕昏仆', '突然昏倒', '突然昏厥', '精神刺激', '肢冷', '脉伏', '脉弦有力', '脉沉细微', '脉细数无力', '脉芤', '自汗', '舌淡', '舌苔薄白', '舌质暗红', '舌质淡', '面色苍白', '面赤'],
    '血厥-虚证': ['不省人事', '口唇无华', '口噤握拳', '呼吸微弱', '呼吸气粗', '唇紫', '四肢厥冷', '四肢震颤', '失血过多', '恐惧', '情志异常', '情绪紧张', '汗出肢冷', '沉弦', '牙关紧闭', '疼痛', '目陷口张', '眩晕昏仆', '突然昏倒', '突然昏厥', '精神刺激', '肢冷', '脉伏', '脉弦有力', '脉沉细微', '脉细数无力', '脉芤', '自汗', '舌淡', '舌苔薄白', '舌质暗红', '舌质淡', '面色苍白', '面赤'],
    '聚证-食滞痰阻': ['便秘', '情绪波动', '纳呆（食欲不佳）', '脉弦', '脉弦滑', '脘胁之间不适', '腹中气聚，攻窜胀痛，时聚时散', '腹痛', '腹胀', '腹部时有条索状物聚起', '舌淡红', '舌苔腻', '苔薄'],
}
# 处理数据
disease_symptoms = process_input_data(input_data)

# 初始化诊断系统
system = DiagnosticSystem(disease_symptoms)

# 第一轮已知症状
current_symptoms = {"胃隐痛", "面色发黄","食欲不振"}

# 获取下一个最优询问症状
next_symptom = system.calculate_next_symptom(current_symptoms)
print(f"建议询问的下一个症状: {next_symptom}")

# 计算当前症状与各疾病的匹配度
scores = system.get_disease_match_scores(current_symptoms)
print("\n当前症状与各疾病的匹配度:")
for disease, score in scores.items():
    print(f"{disease}: {score:.2%}")
