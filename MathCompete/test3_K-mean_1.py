import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional
import os


# =====================
# 数据类定义
# =====================
class MaterialType:
    def __init__(self, id: int, length: float, cost: float, defects: List[Tuple[float, float]]):
        self.id = id
        self.length = length
        self.cost = cost
        self.defects = defects
        self.cluster_label = -1
        self.safe_intervals = self._calculate_safe_intervals()
        self.used_intervals = []

    def _calculate_safe_intervals(self) -> List[Tuple[float, float]]:
        """计算安全切割区间（自动避开缺陷）"""
        intervals = []
        start = 0.0
        for d_start, d_length in sorted(self.defects, key=lambda x: x[0]):
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
# 智能聚类模块
# =====================
class MaterialCluster:
    def __init__(self, max_clusters=3):
        self.max_clusters = max_clusters
        self.scaler = StandardScaler()
        self.kmeans = None

    def fit(self, materials: List[MaterialType]) -> bool:
        """执行聚类并返回是否成功"""
        if len(materials) < 1:
            raise ValueError("至少需要1个原材料样本")

        # 动态调整聚类数
        n_clusters = min(self.max_clusters, len(materials))
        if n_clusters < 1:
            return False

        # 特征工程
        features = []
        for mat in materials:
            defect_pos = min((d[0] for d in mat.defects), default=0)
            defect_total = sum(d[1] for d in mat.defects)
            features.append([
                mat.length,  # 特征1: 总长度
                defect_pos / max(mat.length, 1e-6),  # 避免除以0
                defect_total / max(mat.length, 1e-6)
            ])

        # 标准化和聚类
        X = self.scaler.fit_transform(features)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = self.kmeans.fit_predict(X)

        # 分配标签
        for mat, label in zip(materials, labels):
            mat.cluster_label = label
        return True


# =====================
# 切割优化核心
# =====================
class CuttingOptimizer:
    def __init__(self, kerf_width=0.005):
        self.kerf_width = kerf_width

    def generate_solution(self, materials: List[MaterialType], segments: List[Segment]) -> List[MaterialType]:
        """生成优化后的切割方案"""
        # 按聚类分组
        clusters = {}
        for mat in materials:
            if mat.cluster_label not in clusters:
                clusters[mat.cluster_label] = []
            clusters[mat.cluster_label].append(mat)

        solution = []
        remaining_demand = {seg.id: seg.demand for seg in segments}

        # 按长度降序处理段
        for seg in sorted(segments, key=lambda x: -x.length):
            while remaining_demand[seg.id] > 0:
                placed = False

                # 优先在匹配聚类中分配
                target_cluster = self._select_target_cluster(seg.length)
                for mat in clusters.get(target_cluster, []):
                    if self._try_place_segment(mat, seg):
                        solution.append(mat)
                        remaining_demand[seg.id] -= 1
                        placed = True
                        break

                # 次优选择：其他聚类
                if not placed:
                    for mat in materials:
                        if self._try_place_segment(mat, seg):
                            solution.append(mat)
                            remaining_demand[seg.id] -= 1
                            break

                # 最终无法放置
                if not placed and remaining_demand[seg.id] > 0:
                    print(f"警告: 无法完全放置段ID {seg.id} (剩余{remaining_demand[seg.id]}个)")
                    break

        return solution

    def _select_target_cluster(self, length: float) -> int:
        """根据段长度选择目标聚类"""
        if length > 2.2:
            return 0
        elif length > 1.5:
            return 1
        else:
            return 2

    def _try_place_segment(self, mat: MaterialType, seg: Segment) -> bool:
        """尝试在材料上放置段（考虑锯口和缺陷）"""
        required_length = seg.length + self.kerf_width

        for i, (start, end) in enumerate(mat.safe_intervals):
            if end - start >= required_length:
                # 记录已使用区间
                mat.used_intervals.append((start, start + seg.length))

                # 更新剩余区间
                mat.safe_intervals.pop(i)
                if start + required_length < end:
                    mat.safe_intervals.insert(i, (start + required_length, end))
                return True
        return False


# =====================
# 结果分析
# =====================
class ResultAnalyzer:
    @staticmethod
    def analyze(solution: List[MaterialType], orders: List[Order]):
        # 基本统计
        total_cost = sum(mat.cost for mat in solution)
        total_revenue = sum(o.quantity * o.price for o in orders)

        # 利用率计算
        used_length = sum(
            sum(end - start for (start, end) in mat.used_intervals)
            for mat in solution
        )
        total_length = sum(mat.length for mat in solution)
        utilization = used_length / total_length if total_length > 0 else 0

        print("===== 最终优化结果 =====")
        print(f"总使用原材料: {len(solution)}根")
        print(f"总成本: {total_cost:.2f}元")
        print(f"材料利用率: {utilization * 100:.2f}%")
        print(f"总利润: {total_revenue - total_cost:.2f}元")

        # 聚类统计
        if solution and hasattr(solution[0], 'cluster_label'):
            cluster_stats = {}
            for mat in solution:
                if mat.cluster_label not in cluster_stats:
                    cluster_stats[mat.cluster_label] = []
                cluster_stats[mat.cluster_label].append(mat)

            print("\n===== 聚类使用分析 =====")
            for label, mats in cluster_stats.items():
                avg_len = sum(m.length for m in mats) / len(mats)
                print(f"聚类{label}: 使用{len(mats)}根, 平均长度{avg_len:.2f}m")


# =====================
# 数据加载器
# =====================
class DataLoader:
    @staticmethod
    def load_orders() -> List[Order]:
        return [
            Order("学校教学楼", 1.6, 2.2, 120, 480),
            Order("酒店客房", 1.8, 2.4, 80, 680),
            Order("医院病房", 1.7, 2.3, 60, 550),
            Order("政府办公楼", 1.5, 2.0, 40, 420)
        ]

    @staticmethod
    def load_materials_from_excel(file_path: str) -> List[MaterialType]:
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

            df = pd.read_excel(file_path)
            if len(df) < 3:
                print("警告: 原材料数据少于3条，聚类效果可能受限")

            materials = []
            for _, row in df.iterrows():
                # 处理可能的列名差异
                defect_pos = row.get('缺陷位置', row.get('缺陷位置 (米)', 0))
                defect_len = row.get('缺陷长度', row.get('缺陷长度 (米)', 0))
                defects = [(defect_pos, defect_len)] if defect_pos > 0 or defect_len > 0 else []

                materials.append(MaterialType(
                    row.get('原材料编号', row.get('ID', _)),
                    row.get('长度', row.get('原材料长度 (米)', 0)),
                    row.get('单价', row.get('单价（元/根）', 0)),
                    defects
                ))
            return materials
        except Exception as e:
            raise ValueError(f"Excel加载失败: {str(e)}")


# =====================
# 主流程
# =====================
def main():
    try:
        # 1. 设置文件路径（使用原始字符串处理Windows路径）
        excel_path = r"C:\Users\17813\Desktop\数模\B\附件.xlsx"

        # 2. 加载数据
        orders = DataLoader.load_orders()
        materials = DataLoader.load_materials_from_excel(excel_path)

        # 3. 构建切割需求 (1200段)
        segments = []
        seg_id = 0
        for order in orders:
            segments.append(Segment(seg_id, order.width, order.quantity * 2, "width"))
            seg_id += 1
            segments.append(Segment(seg_id, order.height, order.quantity * 2, "height"))
            seg_id += 1

        # 4. 执行聚类
        cluster = MaterialCluster(max_clusters=3)
        if not cluster.fit(materials):
            print("警告: 聚类未执行，使用原始材料顺序")

        # 5. 优化切割
        optimizer = CuttingOptimizer(kerf_width=0.005)
        solution = optimizer.generate_solution(materials, segments)

        # 6. 分析结果
        ResultAnalyzer.analyze(solution, orders)

    except Exception as e:
        print(f"程序出错: {str(e)}")
        # 显示当前工作目录帮助调试
        print(f"当前工作目录: {os.getcwd()}")


if __name__ == "__main__":
    main()