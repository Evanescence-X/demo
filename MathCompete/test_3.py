import numpy as np
import random
import pandas as pd
import os
from typing import List, Tuple, Dict


# =====================
# 数据定义
# =====================

class MaterialType:
    def __init__(self, id: int, length: float, cost: float, defects: List[Tuple[float, float]]):
        self.id = id
        self.length = length
        self.cost = cost
        self.defects = defects
        self.safe_intervals = self.calculate_safe_intervals()

    def calculate_safe_intervals(self) -> List[Tuple[float, float]]:
        """计算安全区间（避开缺陷）"""
        intervals = []
        start = 0.0

        # 按起始位置排序缺陷
        sorted_defects = sorted(self.defects, key=lambda d: d[0])

        for d_start, d_length in sorted_defects:
            d_end = d_start + d_length
            if start < d_start:
                intervals.append((start, d_start))
            start = d_end

        if start < self.length:
            intervals.append((start, self.length))

        return intervals


class Segment:
    def __init__(self, id: int, length: float, demand: int, segment_type: str):
        self.id = id
        self.length = length
        self.demand = demand
        self.type = segment_type  # 'width' or 'height'


class Order:
    def __init__(self, name: str, width: float, height: float, quantity: int, price: float):
        self.name = name
        self.width = width
        self.height = height
        self.quantity = quantity
        self.price = price


# =====================
# 问题数据初始化 (第三问)
# =====================

def load_materials_from_excel(file_path: str) -> List[MaterialType]:
    """从Excel加载材料数据"""
    try:
        df = pd.read_excel(io=file_path, sheet_name='Sheet1')
        materials = []

        # 按原材料编号分组
        grouped = df.groupby(['原材料编号', '原材料长度 (米)'])

        for (mat_id, length), group in grouped:
            defects = []
            for _, row in group.iterrows():
                defects.append((row['缺陷位置 (米)'], row['缺陷长度 (米)']))

            # 计算平均成本
            avg_cost = group['单价（元/根）'].mean()
            materials.append(MaterialType(mat_id, length, avg_cost, defects))

        return materials
    except Exception as e:
        print(f"加载Excel文件失败: {e}")
        raise


# 定义订单 (第三问)
orders = [
    Order("学校教学楼", 1.6, 2.2, 120, 480),
    Order("酒店客房", 1.8, 2.4, 80, 680),
    Order("医院病房", 1.7, 2.3, 60, 550),
    Order("政府办公楼", 1.5, 2.0, 40, 420)
]

# 锯口宽度
KERF_WIDTH = 0.005

# 加载原材料数据 - 使用绝对路径
excel_path = r"C:\Users\17813\Desktop\数模\B\附件.xlsx"
try:
    material_types = load_materials_from_excel(excel_path)
except Exception as e:
    print(f"错误: {e}")
    print("请检查Excel文件路径和内容是否正确")
    exit()

# 所有需要切割的段
all_segments = []
segment_id = 0

for order in orders:
    # 宽度段需求 (每个窗框需要2个宽度段)
    all_segments.append(Segment(segment_id, order.width, order.quantity * 2, "width"))
    segment_id += 1

    # 高度段需求 (每个窗框需要2个高度段)
    all_segments.append(Segment(segment_id, order.height, order.quantity * 2, "height"))
    segment_id += 1


# =====================
# 贪心算法实现
# =====================

class MaterialInstance:
    """表示一根具体的原材料实例"""

    def __init__(self, material_type: MaterialType, instance_id: int):
        self.material_type = material_type
        self.instance_id = instance_id
        self.cut_segments = []  # 存储切割方案 (position, segment)
        self.used_intervals = []  # 已使用的区间
        self.available_intervals = material_type.safe_intervals.copy()  # 可用安全区间

    def try_place_segment(self, segment: Segment) -> bool:
        """尝试放置一段到原材料上"""
        best_interval_idx = -1
        best_position = -1
        min_waste = float('inf')

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
    # 按长度降序排序段
    sorted_segments = sorted(segments, key=lambda s: s.length, reverse=True)

    # 已使用的原材料实例
    material_instances = []

    # 需求计数
    demand_remaining = {seg.id: seg.demand for seg in segments}

    # 尝试放置每个段
    for segment in sorted_segments:
        placed = False

        # 尝试在已有原材料上放置
        for instance in material_instances:
            if demand_remaining[segment.id] > 0 and instance.try_place_segment(segment):
                demand_remaining[segment.id] -= 1
                placed = True
                break

        # 如果未放置，创建新的原材料实例
        if not placed and demand_remaining[segment.id] > 0:
            # 选择成本最低的原材料类型
            best_material = min(materials, key=lambda m: m.cost)
            new_instance = MaterialInstance(best_material, len(material_instances) + 1)

            if new_instance.try_place_segment(segment):
                material_instances.append(new_instance)
                demand_remaining[segment.id] -= 1

    return material_instances


# =====================
# 遗传算法实现
# =====================

class GeneticAlgorithm:
    def __init__(self, segments, material_types, pop_size=50, elite_size=10, mutation_rate=0.1, generations=100):
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

        # 添加贪心解
        greedy_instances = generate_greedy_solution(self.material_types, self.segments)
        greedy_chromosome = self.solution_to_chromosome(greedy_instances)
        population.append(greedy_chromosome)

        # 添加随机解
        for _ in range(self.pop_size - 1):
            chrom = list(range(len(self.segments)))
            random.shuffle(chrom)
            population.append(chrom)

        return population

    def solution_to_chromosome(self, instances: List[MaterialInstance]) -> List[int]:
        """将解决方案转换为染色体"""
        chrom = []
        for instance in instances:
            for pos, seg in instance.cut_segments:
                chrom.append(seg.id)
        return chrom

    def chromosome_to_solution(self, chromosome: List[int]) -> List[MaterialInstance]:
        """将染色体解码为解决方案"""
        # 创建段ID到对象的映射
        segment_map = {seg.id: seg for seg in self.segments}

        # 解码器状态
        material_instances = []
        current_instance = None
        remaining_demand = {seg.id: seg.demand for seg in self.segments}

        # 按染色体顺序处理每个段
        for seg_id in chromosome:
            segment = segment_map[seg_id]

            # 如果需求已满足，跳过
            if remaining_demand[seg_id] <= 0:
                continue

            placed = False

            # 尝试在现有实例上放置
            if current_instance is not None:
                if current_instance.try_place_segment(segment):
                    remaining_demand[seg_id] -= 1
                    placed = True

            # 尝试在其他实例上放置
            if not placed:
                for instance in material_instances:
                    if instance.try_place_segment(segment):
                        remaining_demand[seg_id] -= 1
                        placed = True
                        current_instance = instance
                        break

            # 创建新实例
            if not placed:
                # 选择成本最低的材料类型
                best_material = min(self.material_types, key=lambda m: m.cost)
                new_instance = MaterialInstance(best_material, len(material_instances) + 1)

                if new_instance.try_place_segment(segment):
                    material_instances.append(new_instance)
                    remaining_demand[seg_id] -= 1
                    current_instance = new_instance

        return material_instances

    def fitness(self, chromosome: List[int]) -> float:
        """计算染色体适应度（成本越低，适应度越高）"""
        solution = self.chromosome_to_solution(chromosome)
        total_cost = sum(inst.material_type.cost for inst in solution)
        return 1.0 / total_cost  # 成本越低，适应度越高

    def select_parents(self):
        """锦标赛选择父代"""
        parents = []
        tournament_size = 3

        for _ in range(2):  # 选择两个父代
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=self.fitness)
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
                while parent2[parent2_ptr] in child:
                    parent2_ptr = (parent2_ptr + 1) % size
                child[i] = parent2[parent2_ptr]
                parent2_ptr = (parent2_ptr + 1) % size

        return child

    def mutate(self, chromosome):
        """交换变异"""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome

    def evolve(self):
        """执行遗传算法进化"""
        for gen in range(self.generations):
            # 评估种群
            graded = [(self.fitness(chrom), chrom) for chrom in self.population]
            graded.sort(key=lambda x: x[0], reverse=True)

            # 选择精英
            elites = [chrom for _, chrom in graded[:self.elite_size]]

            # 创建新一代
            new_population = elites

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
            print(f"Generation {gen + 1}/{self.generations}, Best Fitness: {best_fitness:.6f}")

        # 返回最佳解
        best_chrom = max(self.population, key=self.fitness)
        return self.chromosome_to_solution(best_chrom)


# =====================
# 结果计算与输出
# =====================

def calculate_utilization(solution: List[MaterialInstance]) -> Tuple[float, float]:
    """计算利用率和损失率"""
    total_material = sum(inst.material_type.length for inst in solution)
    total_segment_length = 0.0

    for instance in solution:
        for pos, seg in instance.cut_segments:
            total_segment_length += seg.length

    utilization = total_segment_length / total_material
    loss_rate = 1 - utilization

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


def main():
    # 步骤1: 贪心算法生成初始解
    print("生成贪心算法初始解...")
    greedy_result = generate_greedy_solution(material_types, all_segments)

    # 计算贪心解的成本和利用率
    greedy_cost = sum(inst.material_type.cost for inst in greedy_result)
    greedy_util, greedy_loss = calculate_utilization(greedy_result)

    print(f"贪心算法结果: 成本={greedy_cost}元, 利用率={greedy_util * 100:.2f}%, 损失率={greedy_loss * 100:.2f}%")

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