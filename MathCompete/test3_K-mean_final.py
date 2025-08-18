import numpy as np
import random
from typing import List, Tuple, Dict


# =====================
# 数据定义
# =====================

class MaterialType:
    """原材料类型定义"""

    def __init__(self, id: int, length: float, cost: float, defects: List[Tuple[float, float]]):
        """
        初始化原材料类型
        :param id: 材料ID
        :param length: 材料长度(米)
        :param cost: 材料成本(元)
        :param defects: 缺陷区域列表，每个缺陷表示为(start, end)元组
        """
        self.id = id
        self.length = length
        self.cost = cost
        self.defects = defects
        self.safe_intervals = self.calculate_safe_intervals()

    def calculate_safe_intervals(self) -> List[Tuple[float, float]]:
        """计算安全区间（避开缺陷）"""
        intervals = []
        start = 0.0  # 从材料起点开始

        # 按起始位置排序缺陷
        sorted_defects = sorted(self.defects, key=lambda d: d[0])

        for d_start, d_end in sorted_defects:
            if start < d_start:
                # 缺陷前有安全区间
                intervals.append((start, d_start))
            start = max(start, d_end)  # 移动到缺陷结束位置之后

        # 添加最后一个缺陷后的区间（如果有）
        if start < self.length:
            intervals.append((start, self.length))

        return intervals


class Segment:
    """窗框段定义"""

    def __init__(self, id: int, length: float, demand: int, segment_type: str):
        """
        初始化窗框段
        :param id: 段ID
        :param length: 段长度(米)
        :param demand: 需求数量
        :param segment_type: 段类型('width'或'height')
        """
        self.id = id
        self.length = length
        self.demand = demand
        self.type = segment_type


class Order:
    """客户订单定义"""

    def __init__(self, name: str, width: float, height: float, quantity: int, price: float):
        """
        初始化客户订单
        :param name: 订单名称
        :param width: 窗框宽度(米)
        :param height: 窗框高度(米)
        :param quantity: 窗框数量
        :param price: 单价(元)
        """
        self.name = name
        self.width = width
        self.height = height
        self.quantity = quantity
        self.price = price


# =====================
# 问题数据初始化
# =====================

# 定义原材料类型（3种不同规格的原材料）
material_types = [
    MaterialType(1, 5.5, 18, [(1.0, 1.03), (2.5, 2.54)]),  # 类型1：长5.5m，成本18元，有2处缺陷
    MaterialType(2, 6.2, 22, [(0.5, 0.52), (1.8, 1.85)]),  # 类型2：长6.2m，成本22元，有2处缺陷
    MaterialType(3, 7.8, 28, [(3.0, 3.03)])  # 类型3：长7.8m，成本28元，有1处缺陷
]

# 定义客户订单（4个不同客户的订单）
orders = [
    Order("学校教学楼", 1.6, 2.2, 10, 480),  # 学校：宽1.6m，高2.2m，10个，单价480元
    Order("酒店客房", 1.8, 2.4, 20, 680),  # 酒店：宽1.8m，高2.4m，20个，单价680元
    Order("医院病房", 1.7, 2.3, 20, 550),  # 医院：宽1.7m，高2.3m，20个，单价550元
    Order("政府办公楼", 1.5, 2.0, 15, 420)  # 政府：宽1.5m，高2.0m，15个，单价420元
]

# 锯口宽度（切割时的损耗）
KERF_WIDTH = 0.005  # 5毫米


# =====================
# 需求计算
# =====================

def calculate_segment_demands(orders: List[Order]) -> List[Segment]:
    """
    计算所有窗框段的需求
    每个窗框需要：
    - 2个宽度段（左右边框）
    - 2个高度段（上下边框）
    """
    segments = []
    segment_id = 0  # 段ID计数器

    for order in orders:
        # 宽度段需求 (每个窗框需要2个宽度段)
        segments.append(Segment(segment_id, order.width, order.quantity * 2, "width"))
        segment_id += 1

        # 高度段需求 (每个窗框需要2个高度段)
        segments.append(Segment(segment_id, order.height, order.quantity * 2, "height"))
        segment_id += 1

    return segments


# 所有需要切割的窗框段
all_segments = calculate_segment_demands(orders)


# =====================
# 贪心算法实现
# =====================

class MaterialInstance:
    """表示一根具体的原材料实例"""

    def __init__(self, material_type: MaterialType, instance_id: int):
        """
        初始化原材料实例
        :param material_type: 原材料类型
        :param instance_id: 实例ID
        """
        self.material_type = material_type
        self.instance_id = instance_id
        self.cut_segments = []  # 存储切割方案 (position, segment)
        self.used_intervals = []  # 已使用的区间
        self.available_intervals = material_type.safe_intervals.copy()  # 可用安全区间

    def try_place_segment(self, segment: Segment) -> bool:
        """尝试放置一段到原材料上"""
        best_interval_idx = -1
        best_position = -1
        min_waste = float('inf')  # 最小化浪费

        # 在所有可用区间中寻找最佳位置
        for idx, (start, end) in enumerate(self.available_intervals):
            # 计算可用长度
            available_length = end - start

            # 如果需要与前一锯口保持距离
            if self.cut_segments:
                last_segment_end = self.cut_segments[-1][0] + self.cut_segments[-1][1].length
                if start < last_segment_end:
                    # 同一区间内需要锯口
                    required_length = segment.length + KERF_WIDTH
                else:
                    required_length = segment.length
            else:
                required_length = segment.length

            # 检查是否能放下
            if available_length >= required_length:
                # 计算浪费空间
                waste = available_length - segment.length

                # 如果是同一区间内，需要减去锯口
                if self.cut_segments and start < self.cut_segments[-1][0] + self.cut_segments[-1][1].length:
                    waste -= KERF_WIDTH

                if waste < min_waste:
                    min_waste = waste
                    best_interval_idx = idx
                    best_position = start

        # 如果找到合适位置
        if best_interval_idx != -1:
            # 放置段
            self.cut_segments.append((best_position, segment))

            # 更新区间
            interval_start, interval_end = self.available_intervals[best_interval_idx]
            new_interval_start = best_position + segment.length

            # 如果是同一区间内，需要添加锯口
            if self.cut_segments and len(self.cut_segments) > 1:
                last_segment_end = self.cut_segments[-2][0] + self.cut_segments[-2][1].length
                if best_position < last_segment_end:
                    new_interval_start += KERF_WIDTH

            # 更新或删除区间
            if new_interval_start < interval_end:
                self.available_intervals[best_interval_idx] = (new_interval_start, interval_end)
            else:
                del self.available_intervals[best_interval_idx]

            return True

        return False


def generate_greedy_solution(materials: List[MaterialType], segments: List[Segment]) -> List[MaterialInstance]:
    """贪心算法生成初始解"""
    # 按长度降序排序段（优先放置长段）
    sorted_segments = sorted(segments, key=lambda s: s.length, reverse=True)

    # 已使用的原材料实例
    material_instances = []

    # 需求计数
    demand_remaining = {seg.id: seg.demand for seg in segments}

    # 尝试放置每个段
    for segment in sorted_segments:
        while demand_remaining[segment.id] > 0:
            placed = False

            # 尝试在已有原材料上放置
            for instance in material_instances:
                if instance.try_place_segment(segment):
                    demand_remaining[segment.id] -= 1
                    placed = True
                    break

            # 如果未放置，创建新的原材料实例
            if not placed:
                # 选择成本最低的原材料类型
                best_material = min(materials, key=lambda m: m.cost)
                new_instance = MaterialInstance(best_material, len(material_instances) + 1)

                if new_instance.try_place_segment(segment):
                    material_instances.append(new_instance)
                    demand_remaining[segment.id] -= 1
                else:
                    # 如果无法放置，选择更大的材料
                    larger_materials = [m for m in materials if m.length > best_material.length]
                    if larger_materials:
                        best_material = min(larger_materials, key=lambda m: m.cost)
                        new_instance = MaterialInstance(best_material, len(material_instances) + 1)
                        if new_instance.try_place_segment(segment):
                            material_instances.append(new_instance)
                            demand_remaining[segment.id] -= 1
                    else:
                        raise ValueError(f"无法放置段 {segment.id}，没有足够大的材料")

    return material_instances


# =====================
# 遗传算法实现
# =====================

class GeneticAlgorithm:
    """遗传算法优化切割方案"""

    def __init__(self, segments, material_types, pop_size=50, elite_size=10, mutation_rate=0.1, generations=100):
        """
        初始化遗传算法
        :param segments: 所有需要切割的段
        :param material_types: 原材料类型
        :param pop_size: 种群大小
        :param elite_size: 精英数量
        :param mutation_rate: 变异率
        :param generations: 迭代次数
        """
        self.segments = segments
        self.material_types = material_types
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations

        # 创建初始种群
        self.population = self.initial_population()

    def initial_population(self):
        """创建初始种群"""
        population = []

        # 添加贪心解（保证初始解质量）
        greedy_instances = generate_greedy_solution(self.material_types, self.segments)
        greedy_chromosome = self.solution_to_chromosome(greedy_instances)
        population.append(greedy_chromosome)

        # 添加随机解（增加多样性）
        for _ in range(self.pop_size - 1):
            # 生成满足需求的随机染色体
            chrom = []
            for seg in self.segments:
                chrom.extend([seg.id] * seg.demand)
            random.shuffle(chrom)
            population.append(chrom)

        return population

    def solution_to_chromosome(self, instances: List[MaterialInstance]) -> List[int]:
        """将解决方案转换为染色体（段ID序列）"""
        chrom = []
        for instance in instances:
            for pos, seg in instance.cut_segments:
                chrom.append(seg.id)
        return chrom

    def chromosome_to_solution(self, chromosome: List[int]) -> List[MaterialInstance]:
        """将染色体解码为解决方案 - 确保满足所有需求"""
        segment_map = {seg.id: seg for seg in self.segments}
        material_instances = []
        remaining_demand = {seg.id: seg.demand for seg in self.segments}

        # 处理染色体中的每个段
        for seg_id in chromosome:
            if remaining_demand[seg_id] <= 0:
                continue

            segment = segment_map[seg_id]
            placed = False

            # 尝试在现有材料上放置
            for instance in material_instances:
                if instance.try_place_segment(segment):
                    remaining_demand[seg_id] -= 1
                    placed = True
                    break

            # 如果无法放置，创建新材料
            if not placed:
                best_material = min(self.material_types, key=lambda m: m.cost)
                new_instance = MaterialInstance(best_material, len(material_instances) + 1)
                if new_instance.try_place_segment(segment):
                    material_instances.append(new_instance)
                    remaining_demand[seg_id] -= 1

        # 检查是否还有未满足的需求
        for seg_id, demand in remaining_demand.items():
            if demand > 0:
                # 为剩余需求创建新材料
                segment = segment_map[seg_id]
                while remaining_demand[seg_id] > 0:
                    best_material = min(self.material_types, key=lambda m: m.cost)
                    new_instance = MaterialInstance(best_material, len(material_instances) + 1)
                    if new_instance.try_place_segment(segment):
                        material_instances.append(new_instance)
                        remaining_demand[seg_id] -= 1

        return material_instances

    def fitness(self, chromosome: List[int]) -> float:
        """计算染色体适应度（成本越低，适应度越高）"""
        solution = self.chromosome_to_solution(chromosome)
        total_cost = sum(inst.material_type.cost for inst in solution)
        return 1.0 / (total_cost + 1)  # 使用倒数使成本越低适应度越高，+1避免除零错误

    def select_parents(self):
        """锦标赛选择父代"""
        parents = []
        tournament_size = 3  # 锦标赛大小

        for _ in range(2):  # 选择两个父代
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=self.fitness)  # 选择适应度最高的
            parents.append(winner)

        return parents

    def crossover(self, parent1, parent2):
        """顺序交叉（OX）"""
        size = len(parent1)

        # 选择两个交叉点
        idx1, idx2 = sorted(random.sample(range(size), 2))

        # 创建子代
        child = [-1] * size

        # 从parent1复制片段
        child[idx1:idx2] = parent1[idx1:idx2]

        # 从parent2填充剩余位置
        parent2_ptr = 0
        for i in range(size):
            if child[i] == -1:  # 空位
                while parent2[parent2_ptr] in child[idx1:idx2]:
                    parent2_ptr = (parent2_ptr + 1) % size
                child[i] = parent2[parent2_ptr]
                parent2_ptr = (parent2_ptr + 1) % size

        return child

    def mutate(self, chromosome):
        """交换变异"""
        if random.random() < self.mutation_rate and len(chromosome) >= 2:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome

    def evolve(self):
        """执行遗传算法进化"""
        for gen in range(self.generations):
            # 评估种群
            graded = [(self.fitness(chrom), chrom) for chrom in self.population]
            graded.sort(key=lambda x: x[0], reverse=True)  # 按适应度降序排序

            # 选择精英（直接保留到下一代）
            elites = [chrom for _, chrom in graded[:self.elite_size]]

            # 创建新一代
            new_population = elites.copy()

            # 生成后代
            while len(new_population) < self.pop_size:
                # 选择父代
                parent1, parent2 = self.select_parents()

                # 交叉
                child = self.crossover(parent1, parent2)

                # 变异
                child = self.mutate(child)

                new_population.append(child)

            self.population = new_population

            # 打印进度
            best_fitness = graded[0][0]
            best_cost = 1 / best_fitness - 1
            print(f"Generation {gen + 1}/{self.generations}, Best Cost: {best_cost:.2f}")

        # 返回最佳解
        best_chrom = max(self.population, key=self.fitness)
        return self.chromosome_to_solution(best_chrom)


# =====================
# 结果计算与输出
# =====================

def calculate_utilization(solution: List[MaterialInstance]) -> Tuple[float, float]:
    """计算材料利用率和损失率"""
    total_material = sum(inst.material_type.length for inst in solution)
    total_segment_length = 0.0

    for instance in solution:
        for pos, seg in instance.cut_segments:
            total_segment_length += seg.length

    utilization = total_segment_length / total_material  # 利用率 = 使用长度/总长度
    loss_rate = 1 - utilization  # 损失率 = 1 - 利用率

    return utilization, loss_rate


def print_solution(solution: List[MaterialInstance]):
    """打印切割方案"""
    print("\n===== 最优切割方案 =====")
    total_cost = 0
    total_segments = 0

    for instance in solution:
        mat_type = instance.material_type
        print(f"\n原材料 {instance.instance_id} (类型 {mat_type.id}, 长度 {mat_type.length}m, 成本 {mat_type.cost}元):")

        # 按位置排序段
        segments_sorted = sorted(instance.cut_segments, key=lambda x: x[0])

        for i, (pos, seg) in enumerate(segments_sorted):
            seg_type = "宽度" if seg.type == "width" else "高度"
            print(f"  段 {i + 1}: 位置 {pos:.3f}m, 长度 {seg.length}m ({seg_type})")

        total_cost += mat_type.cost
        total_segments += len(segments_sorted)

    print(f"\n总使用原材料数: {len(solution)}根")
    print(f"总切割段数: {total_segments}段")
    print(f"总成本: {total_cost}元")


def verify_solution(solution: List[MaterialInstance], segments: List[Segment]):
    """验证解决方案是否满足所有需求"""
    total_required = {seg.id: seg.demand for seg in segments}
    total_cut = {seg.id: 0 for seg in segments}

    for instance in solution:
        for pos, seg in instance.cut_segments:
            total_cut[seg.id] += 1

    all_satisfied = True
    for seg_id, demand in total_required.items():
        if total_cut[seg_id] < demand:
            print(f"❌ 段{seg_id}需求{demand}，实际切割{total_cut[seg_id]}")
            all_satisfied = False
        elif total_cut[seg_id] > demand:
            print(f"⚠️ 段{seg_id}需求{demand}，实际切割{total_cut[seg_id]} (超出需求)")

    if all_satisfied:
        print("✅ 所有需求均已满足！")
    else:
        print("❌ 存在未满足的需求")


def main():
    """主函数"""
    # 步骤1: 贪心算法生成初始解
    print("生成贪心算法初始解...")
    greedy_result = generate_greedy_solution(material_types, all_segments)

    # 计算贪心解的成本和利用率
    greedy_cost = sum(inst.material_type.cost for inst in greedy_result)
    greedy_util, greedy_loss = calculate_utilization(greedy_result)

    print(f"贪心算法结果: 成本={greedy_cost}元, 利用率={greedy_util * 100:.2f}%, 损失率={greedy_loss * 100:.2f}%")
    verify_solution(greedy_result, all_segments)

    # 步骤2: 遗传算法优化
    print("\n启动遗传算法优化...")
    ga = GeneticAlgorithm(
        segments=all_segments,
        material_types=material_types,
        pop_size=50,
        elite_size=10,
        mutation_rate=0.1,
        generations=100
    )

    best_solution = ga.evolve()

    # 计算最终结果
    total_cost = sum(inst.material_type.cost for inst in best_solution)
    utilization, loss_rate = calculate_utilization(best_solution)

    # 步骤3: 输出结果
    print("\n===== 最终优化结果 =====")
    print(f"总成本: {total_cost}元")
    print(f"材料利用率: {utilization * 100:.2f}%")
    print(f"切割损失率: {loss_rate * 100:.2f}%")

    print_solution(best_solution)
    verify_solution(best_solution, all_segments)

    # 步骤4: 计算利润
    total_revenue = sum(
        order.quantity * order.price
        for order in orders
    )
    profit = total_revenue - total_cost

    print(f"\n总收益: {total_revenue}元")
    print(f"总利润: {profit}元")


if __name__ == "__main__":
    main()