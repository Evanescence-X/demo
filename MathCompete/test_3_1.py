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
    Order("学校教学楼", 1.6, 2.2, 120, 480),  # 120套，每套需要2宽2高
    Order("酒店客房", 1.8, 2.4, 80, 680),  # 80套
    Order("医院病房", 1.7, 2.3, 60, 550),  # 60套
    Order("政府办公楼", 1.5, 2.0, 40, 420)  # 40套
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

# 打印需求汇总

print("=== 切割需求汇总 ===")
total_frames = sum(order.quantity for order in orders)
total_width_segments = sum(order.quantity * 2 for order in orders)
total_height_segments = sum(order.quantity * 2 for order in orders)
print(f"总窗框数: {total_frames}套")
print(f"总宽度段需求: {total_width_segments}根 (每种窗框2根)")
print(f"总高度段需求: {total_height_segments}根 (每种窗框2根)")

# [后续的MaterialInstance类、generate_greedy_solution函数、GeneticAlgorithm类等实现保持不变]
# [只需确保上面的需求计算正确即可]