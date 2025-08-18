import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Dict, Optional
import os
import matplotlib.pyplot as plt


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
# 智能聚类模块（改进版）
# =====================
class MaterialCluster:
    def __init__(self, max_clusters=5):
        self.max_clusters = max_clusters
        self.scaler = StandardScaler()
        self.kmeans = None

    def fit(self, materials: List[MaterialType]) -> bool:
        """执行聚类（支持自动K值选择）"""
        if len(materials) < 1:
            raise ValueError("至少需要1个原材料样本")

        # 单样本特殊情况
        if len(materials) == 1:
            materials[0].cluster_label = 0
            return True

        # 提取增强特征
        X = self._extract_features(materials)
        X_scaled = self.scaler.fit_transform(X)

        # 自动确定最佳K值
        n_clusters = self._auto_select_k(X_scaled)
        print(f"最终使用聚类数: {n_clusters}")

        # 执行聚类
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = self.kmeans.fit_predict(X_scaled)

        # 分配标签
        for mat, label in zip(materials, labels):
            mat.cluster_label = label

        # 可视化分析
        self._plot_cluster_stats(materials, X_scaled)
        return True

    def _extract_features(self, materials):
        """提取增强后的特征（6维特征向量）"""
        features = []
        for mat in materials:
            defects = sorted(mat.defects, key=lambda x: x[0])
            defect_positions = [d[0] for d in defects]
            defect_lengths = [d[1] for d in defects]
            safe_intervals = mat.safe_intervals

            features.append([
                mat.length,  # 特征1: 总长度
                sum(defect_lengths) / max(mat.length, 1e-6),  # 特征2: 缺陷总占比
                np.mean(defect_positions) / max(mat.length, 1e-6) if defects else 0,  # 特征3: 平均缺陷位置
                len(defects),  # 特征4: 缺陷数量
                np.std(defect_lengths) if len(defects) > 1 else 0,  # 特征5: 缺陷长度变异
                np.mean([e - s for s, e in safe_intervals]) if safe_intervals else 0  # 特征6: 安全区间平均长度
            ])
        return np.array(features)

    def _auto_select_k(self, X_scaled, max_k=8):
        """通过轮廓系数自动选择最佳K值"""
        possible_k = range(2, min(max_k, len(X_scaled)))
        silhouette_scores = []

        for k in possible_k:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
            print(f"K={k} 轮廓系数: {score:.3f}")

        best_k = possible_k[np.argmax(silhouette_scores)]
        return min(best_k, self.max_clusters)  # 不超过用户设定的上限

    def _plot_cluster_stats(self, materials, X_scaled):
        """绘制聚类分析图"""
        plt.figure(figsize=(15, 5))

        # 特征重要性热力图
        plt.subplot(131)
        corr = np.corrcoef(X_scaled.T)
        plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("Feature Correlation")

        # 长度-缺陷数量散点图
        plt.subplot(132)
        for mat in materials:
            plt.scatter(mat.length, len(mat.defects), c=mat.cluster_label, alpha=0.6)
        plt.xlabel("Length")
        plt.ylabel("Defect Count")
        plt.title("Length vs Defects")

        # 各聚类材料数量
        plt.subplot(133)
        labels = [mat.cluster_label for mat in materials]
        plt.hist(labels, bins=len(set(labels)), rwidth=0.8)
        plt.xlabel("Cluster Label")
        plt.ylabel("Count")
        plt.title("Materials per Cluster")

        plt.tight_layout()
        plt.show()


# =====================
# 切割优化核心（适配多聚类）
# =====================
class CuttingOptimizer:
    def __init__(self, kerf_width=0.005):
        self.kerf_width = kerf_width

    def generate_solution(self, materials: List[MaterialType], segments: List[Segment]) -> List[MaterialType]:
        """生成优化后的切割方案（支持多聚类）"""
        # 按聚类分组并排序（优先使用长材料）
        clusters = {}
        for mat in materials:
            if mat.cluster_label not in clusters:
                clusters[mat.cluster_label] = []
            clusters[mat.cluster_label].append(mat)

        # 每个聚类按长度降序排列
        for label in clusters:
            clusters[label].sort(key=lambda x: -x.length)

        solution = []
        remaining_demand = {seg.id: seg.demand for seg in segments}

        # 按长度降序处理段
        for seg in sorted(segments, key=lambda x: -x.length):
            while remaining_demand[seg.id] > 0:
                placed = False

                # 1. 优先在目标聚类中分配
                target_cluster = self._select_target_cluster(seg.length)
                if target_cluster in clusters:
                    for mat in clusters[target_cluster]:
                        if self._try_place_segment(mat, seg):
                            solution.append(mat)
                            remaining_demand[seg.id] -= 1
                            placed = True
                            break

                # 2. 次优：其他聚类中更长的材料
                if not placed:
                    for label in sorted(clusters.keys(), key=lambda x: abs(x - target_cluster)):
                        for mat in clusters[label]:
                            if self._try_place_segment(mat, seg):
                                solution.append(mat)
                                remaining_demand[seg.id] -= 1
                                placed = True
                                break
                        if placed: break

                # 3. 最终尝试：任何可用材料
                if not placed:
                    for mat in materials:
                        if self._try_place_segment(mat, seg):
                            solution.append(mat)
                            remaining_demand[seg.id] -= 1
                            break

                # 无法放置时的处理
                if not placed and remaining_demand[seg.id] > 0:
                    print(f"警告: 无法放置段ID {seg.id} (长度{seg.length}m，剩余需求{remaining_demand[seg.id]})")
                    break

        return solution

    def _select_target_cluster(self, length: float) -> int:
        """动态选择目标聚类（适配多聚类）"""
        if length > 2.5:
            return 0  # 超长段专用
        elif length > 2.0:
            return 1  # 长段优先
        elif length > 1.5:
            return 2  # 中长段
        elif length > 1.0:
            return 3  # 短段
        else:
            return 4  # 边角料

    def _try_place_segment(self, mat: MaterialType, seg: Segment) -> bool:
        """尝试放置段（考虑锯口和缺陷）"""
        required_length = seg.length + self.kerf_width

        for i, (start, end) in enumerate(mat.safe_intervals):
            if end - start >= required_length:
                # 记录已使用区间
                mat.used_intervals.append((start, start + seg.length))

                # 更新剩余区间
                remaining = end - (start + required_length)
                mat.safe_intervals.pop(i)
                if remaining > 0.01:  # 忽略小于1cm的余料
                    mat.safe_intervals.insert(i, (start + required_length, end))
                return True
        return False


# =====================
# 结果分析（增强版）
# =====================
class ResultAnalyzer:
    @staticmethod
    def analyze(solution: List[MaterialType], orders: List[Order]):
        # 基本统计
        used_materials = len(solution)
        total_cost = sum(mat.cost for mat in solution)
        total_revenue = sum(o.quantity * o.price for o in orders)

        # 利用率计算
        used_length = sum(sum(end - start for (start, end) in mat.used_intervals) for mat in solution)
        total_length = sum(mat.length for mat in solution)
        utilization = used_length / total_length if total_length > 0 else 0

        # 聚类统计
        cluster_stats = {}
        for mat in solution:
            if mat.cluster_label not in cluster_stats:
                cluster_stats[mat.cluster_label] = {"count": 0, "total_len": 0, "used_len": 0}
            cluster_stats[mat.cluster_label]["count"] += 1
            cluster_stats[mat.cluster_label]["total_len"] += mat.length
            cluster_stats[mat.cluster_label]["used_len"] += sum(e - s for (s, e) in mat.used_intervals)

        print("=" * 40)
        print("优化结果摘要".center(40))
        print("=" * 40)
        print(f"总使用材料: {used_materials}根")
        print(f"总成本: {total_cost:.2f}元")
        print(f"材料利用率: {utilization * 100:.2f}%")
        print(f"预估利润: {total_revenue - total_cost:.2f}元")

        print("\n聚类使用情况:")
        for label, stat in cluster_stats.items():
            avg_len = stat["total_len"] / stat["count"]
            cluster_util = stat["used_len"] / stat["total_len"]
            print(f" 聚类{label}: {stat['count']}根, 平均长度{avg_len:.2f}m, 利用率{cluster_util * 100:.2f}%")

        # 绘制利用率分布图
        plt.figure(figsize=(8, 4))
        utils = [sum(e - s for (s, e) in mat.used_intervals) / mat.length for mat in solution]
        plt.hist(utils, bins=20, range=(0, 1), edgecolor='k')
        plt.xlabel("Utilization Rate")
        plt.ylabel("Count")
        plt.title("Material Utilization Distribution")
        plt.show()


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
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

            df = pd.read_excel(file_path)
            materials = []

            for _, row in df.iterrows():
                # 处理灵活的列名
                id_col = next((col for col in ['ID', '原材料编号', '编号'] if col in df.columns), None)
                len_col = next((col for col in ['长度', '原材料长度', '长度(m)'] if col in df.columns), None)
                cost_col = next((col for col in ['单价', '成本', '单价（元）'] if col in df.columns), None)
                defect_pos_col = next((col for col in ['缺陷位置', '缺陷位置(m)'] if col in df.columns), None)
                defect_len_col = next((col for col in ['缺陷长度', '缺陷长度(m)'] if col in df.columns), None)

                defects = []
                if defect_pos_col and defect_len_col:
                    pos = row[defect_pos_col]
                    length = row[defect_len_col]
                    if pd.notnull(pos) and pd.notnull(length) and length > 0:
                        defects.append((float(pos), float(length)))

                materials.append(MaterialType(
                    row[id_col] if id_col else _,
                    float(row[len_col]) if len_col else 0,
                    float(row[cost_col]) if cost_col else 0,
                    defects
                ))

            print(f"成功加载 {len(materials)} 条原材料数据")
            return materials
        except Exception as e:
            raise ValueError(f"数据加载失败: {str(e)}")


# =====================
# 主流程
# =====================
def main():
    try:
        # 1. 设置文件路径
        excel_path = r"C:\Users\17813\Desktop\数模\B\附件.xlsx"  # 修改为实际路径

        # 2. 加载数据
        orders = DataLoader.load_orders()
        materials = DataLoader.load_materials_from_excel(excel_path)

        # 3. 构建切割需求 (每个订单需要2*quantity个段)
        segments = []
        seg_id = 0
        for order in orders:
            segments.append(Segment(seg_id, order.width, order.quantity * 2, "width"))
            seg_id += 1
            segments.append(Segment(seg_id, order.height, order.quantity * 2, "height"))
            seg_id += 1

        # 4. 执行智能聚类
        print("\n正在进行材料聚类分析...")
        cluster = MaterialCluster(max_clusters=5)  # 可调整最大聚类数
        cluster.fit(materials)

        # 5. 优化切割方案
        print("\n生成切割方案中...")
        optimizer = CuttingOptimizer(kerf_width=0.005)
        solution = optimizer.generate_solution(materials, segments)

        # 6. 分析结果
        print("\n正在分析优化结果...")
        ResultAnalyzer.analyze(solution, orders)

    except Exception as e:
        print(f"程序出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()