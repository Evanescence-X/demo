import numpy as np
import random
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import gc
import sys

# 订单数据
orders = [
    {'id': 1, 'name': '学校教学楼', 'quantity': 10, 'width': 1.6, 'height': 2.2, 'price': 480},
    {'id': 2, 'name': '酒店客房', 'quantity': 20, 'width': 1.8, 'height': 2.4, 'price': 680},
    {'id': 3, 'name': '医院病房', 'quantity': 20, 'width': 1.7, 'height': 2.3, 'price': 550},
    {'id': 4, 'name': '政府办公楼', 'quantity': 15, 'width': 1.5, 'height': 2.0, 'price': 420}
]

# 原材料数据
materials = [
    {'id': 1, 'length': 5.5, 'price': 18, 'defects': [(1.0, 0.03), (2.5, 0.04)]},
    {'id': 2, 'length': 6.2, 'price': 22, 'defects': [(0.5, 0.02), (1.8, 0.05)]},
    {'id': 3, 'length': 7.8, 'price': 28, 'defects': [(3.0, 0.03)]}
]

# 全局参数
SAW_WIDTH = 0.005
TOLERANCE = 0.01
MAX_PARTS_PER_SOLUTION = 200


def get_available_segments(length: float, defects: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """计算考虑缺陷后的可用段"""
    if not defects:
        return [(0, length)]

    # 按起始位置排序缺陷
    defects_sorted = sorted(defects, key=lambda x: x[0])
    available_segments = []
    prev_end = 0

    for defect_start, defect_length in defects_sorted:
        defect_end = defect_start + defect_length
        if defect_start > prev_end:
            available_segments.append((prev_end, defect_start))
        prev_end = max(prev_end, defect_end)

    if prev_end < length:
        available_segments.append((prev_end, length))

    return available_segments


class CuttingSolution:
    def __init__(self):
        self.patterns = []  # (material_id, length, part_type, order_id)
        self.total_cost = 0
        self.total_revenue = 0
        self.total_waste = 0
        self.total_used = 0
        self.fitness = 0

    def calculate_fitness(self):
        self.fitness = self.total_revenue - self.total_cost
        return self.fitness

    def copy(self):
        new_solution = CuttingSolution()
        new_solution.patterns = self.patterns.copy()
        new_solution.total_cost = self.total_cost
        new_solution.total_revenue = self.total_revenue
        new_solution.total_waste = self.total_waste
        new_solution.total_used = self.total_used
        new_solution.fitness = self.fitness
        return new_solution

    def add_pattern(self, material_id, length, part_type, order_id, price):
        if len(self.patterns) < MAX_PARTS_PER_SOLUTION:
            self.patterns.append((material_id, length, part_type, order_id))
            self.total_cost += materials[material_id - 1]['price']
            self.total_revenue += price
            return True
        return False


def safe_concat(list1, list2):
    result = list1.copy()
    result.extend(list2)
    return result


def generate_initial_solution() -> CuttingSolution:
    solution = CuttingSolution()
    parts_needed = []

    for order in orders:
        min_w = order['width'] - TOLERANCE
        max_w = order['width'] + TOLERANCE
        min_h = order['height'] - TOLERANCE
        max_h = order['height'] + TOLERANCE

        for _ in range(2 * order['quantity']):
            parts_needed.append({
                'type': 'width',
                'min': min_w,
                'max': max_w,
                'order_id': order['id'],
                'price': order['price'] / 4
            })
            parts_needed.append({
                'type': 'height',
                'min': min_h,
                'max': max_h,
                'order_id': order['id'],
                'price': order['price'] / 4
            })

    parts_needed.sort(key=lambda x: x['max'], reverse=True)

    while len(parts_needed) > 0:
        best_material = None
        best_remaining_length = float('inf')
        best_used_parts = []

        for material in materials:
            defects = material['defects']
            available_segments = get_available_segments(material['length'], defects)

            used_parts = []
            remaining_segments = available_segments.copy()

            for part in parts_needed[:]:
                part_length = (part['min'] + part['max']) / 2
                total_length = part_length + SAW_WIDTH

                for i, segment in enumerate(remaining_segments):
                    if segment[1] - segment[0] >= total_length:
                        if solution.add_pattern(
                                material_id=material['id'],
                                length=part_length,
                                part_type=part['type'],
                                order_id=part['order_id'],
                                price=part['price'] * 4
                        ):
                            used_parts.append(part)
                            remaining_segments[i] = (segment[0] + total_length, segment[1])
                            parts_needed.remove(part)
                        break

            remaining_length = sum(end - start for start, end in remaining_segments)
            if len(used_parts) > 0 and remaining_length < best_remaining_length:
                best_material = material
                best_remaining_length = remaining_length
                best_used_parts = used_parts

        if not best_used_parts and len(parts_needed) > 0:
            pass

    solution.calculate_fitness()
    return solution


def genetic_algorithm_optimization(population_size=15, generations=30, mutation_rate=0.1):
    if sys.getsizeof([]) * population_size * generations > 100 * 1024 * 1024:
        population_size = min(population_size, 10)
        generations = min(generations, 20)

    population = [generate_initial_solution() for _ in range(population_size)]
    best_solution = max(population, key=lambda x: x.fitness).copy()
    fitness_history = [best_solution.fitness]

    for generation in range(generations):
        gc.collect()

        fitnesses = [sol.fitness for sol in population]
        total_fitness = sum(fitnesses)
        selected = []

        for _ in range(population_size):
            pick = random.uniform(0, total_fitness)
            current = 0
            for sol in population:
                current += sol.fitness
                if current > pick:
                    selected.append(sol.copy())
                    break

        new_population = []
        for i in range(0, population_size, 2):
            if i + 1 >= population_size:
                new_population.append(selected[i].copy())
                continue

            parent1, parent2 = selected[i], selected[i + 1]
            cross_point1 = random.randint(0, len(parent1.patterns) - 1) if parent1.patterns else 0
            cross_point2 = random.randint(0, len(parent2.patterns) - 1) if parent2.patterns else 0

            child1 = CuttingSolution()
            child1.patterns = safe_concat(parent1.patterns[:cross_point1], parent2.patterns[cross_point2:])

            child2 = CuttingSolution()
            child2.patterns = safe_concat(parent2.patterns[:cross_point2], parent1.patterns[cross_point1:])

            for child in [child1, child2]:
                child.total_cost = sum(materials[p[0] - 1]['price'] for p in child.patterns)
                child.total_revenue = sum(orders[p[3] - 1]['price'] for p in child.patterns)
                child.calculate_fitness()

            new_population.extend([child1, child2])

        for sol in new_population:
            if random.random() < mutation_rate and sol.patterns:
                mutation_type = random.choice(['swap', 'remove'])

                if mutation_type == 'swap' and len(sol.patterns) > 1:
                    i, j = random.sample(range(len(sol.patterns)), 2)
                    sol.patterns[i], sol.patterns[j] = sol.patterns[j], sol.patterns[i]

                elif mutation_type == 'remove':
                    idx = random.randint(0, len(sol.patterns) - 1)
                    removed = sol.patterns.pop(idx)
                    sol.total_cost -= materials[removed[0] - 1]['price']
                    sol.total_revenue -= orders[removed[3] - 1]['price']

        population = new_population
        current_best = max(population, key=lambda x: x.fitness)
        if current_best.fitness > best_solution.fitness:
            best_solution = current_best.copy()

        fitness_history.append(best_solution.fitness)

    plt.plot(fitness_history)
    plt.title('Optimized Genetic Algorithm Performance')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (Profit)')
    plt.show()

    return best_solution


def evaluate_solution(solution: CuttingSolution):
    total_material_length = sum(materials[p[0] - 1]['length'] for p in solution.patterns)
    total_used_length = solution.total_used
    total_waste = solution.total_waste

    utilization = total_used_length / total_material_length
    loss_rate = total_waste / total_material_length

    print("\n=== 最优切割方案评估 ===")
    print(f"总利润: {solution.fitness:.2f} 元")
    print(f"总成本: {solution.total_cost:.2f} 元")
    print(f"总收益: {solution.total_revenue:.2f} 元")
    print(f"\n材料使用情况:")
    print(f"总用料长度: {total_material_length:.2f} 米")
    print(f"总利用长度: {total_used_length:.2f} 米")
    print(f"总浪费长度: {total_waste:.2f} 米")
    print(f"材料利用率: {utilization * 100:.2f}%")
    print(f"切割损失率: {loss_rate * 100:.2f}%")

    order_counts = {}
    for order in orders:
        order_counts[order['id']] = {'width': 0, 'height': 0, 'needed': order['quantity'] * 2}

    for pattern in solution.patterns:
        part_type = pattern[2]
        order_id = pattern[3]
        if part_type == 'width':
            order_counts[order_id]['width'] += 1
        else:
            order_counts[order_id]['height'] += 1

    print("\n=== 订单完成情况 ===")
    for order in orders:
        oid = order['id']
        needed = order_counts[oid]['needed']
        width = order_counts[oid]['width']
        height = order_counts[oid]['height']
        completed = min(width, height) // 2
        print(f"订单 {oid}({order['name']}):")
        print(f"  需要: {order['quantity']} 套")
        print(f"  已完成: {completed} 套 (宽度部件: {width}/{needed}, 高度部件: {height}/{needed})")
        print(f"  完成率: {completed / order['quantity'] * 100:.1f}%")


if __name__ == "__main__":
    print("开始运行优化算法...")
    best_solution = genetic_algorithm_optimization()
    evaluate_solution(best_solution)