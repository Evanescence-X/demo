# 导入PuLP库，用于线性规划建模和求解
from pulp import *

# ============== 原材料参数设置 ==============
# 定义原材料的长度（单位：米）
lengths = [5.5, 6.2, 7.8]  # 原材料长度
# 定义原材料的单价（单位：元）
costs = [18, 22, 28]       # 原材料单价
# 定义锯口宽度（单位：米），即切割时损耗的宽度
w = 0.005                 # 锯口宽度

# ============== 订单需求设置 ==============
# 定义宽度段的订单需求，格式：(段名称, 段长度, 需求数量)
width_segments = [
    ('W1', 1.6, 20),  # 订单1宽度段：名称W1，长度1.6米，需求20段
    ('W2', 1.8, 40),  # 订单2宽度段：名称W2，长度1.8米，需求40段
    ('W3', 1.7, 40),  # 订单3宽度段：名称W3，长度1.7米，需求40段
    ('W4', 1.5, 30)   # 订单4宽度段：名称W4，长度1.5米，需求30段
]

# 定义高度段的订单需求，格式：(段名称, 段长度, 需求数量)
height_segments = [
    ('H1', 2.2, 20),  # 订单1高度段：名称H1，长度2.2米，需求20段
    ('H2', 2.4, 40),  # 订单2高度段：名称H2，长度2.4米，需求40段
    ('H3', 2.3, 40),  # 订单3高度段：名称H3，长度2.3米，需求40段
    ('H4', 2.0, 30)   # 订单4高度段：名称H4，长度2.0米，需求30段
]

# 合并宽度段和高度段的需求列表
segments = width_segments + height_segments

# ============== 计算最大切割段数 ==============
# 定义k_max字典，存储每种原材料切割每种段的最大可能数量
k_max = {}

# 第一根原材料：5.5米
k_max[('5.5', 'W1')] = 3  # 1.6*3 + 2*0.005=4.81 <=5.5
k_max[('5.5', 'W2')] = 3  # 1.8*3 + 2*0.005=5.41 <=5.5
k_max[('5.5', 'W3')] = 3  # 1.7*3 + 2*0.005=5.11 <=5.5
k_max[('5.5', 'W4')] = 3  # 1.5*3 + 2*0.005=4.51 <=5.5
k_max[('5.5', 'H1')] = 2  # 2.2*2 + 1*0.005=4.405 <=5.5
k_max[('5.5', 'H2')] = 2  # 2.4*2 + 1*0.005=4.805 <=5.5
k_max[('5.5', 'H3')] = 2  # 2.3*2 + 1*0.005=4.605 <=5.5
k_max[('5.5', 'H4')] = 2  # 2.0*2 + 1*0.005=4.005 <=5.5

# 第二根原材料：6.2米
k_max[('6.2', 'W1')] = 3  # 1.6*3 + 2*0.005=4.81 <=6.2
k_max[('6.2', 'W2')] = 3  # 1.8*3 + 2*0.005=5.41 <=6.2
k_max[('6.2', 'W3')] = 3  # 1.7*3 + 2*0.005=5.11 <=6.2
k_max[('6.2', 'W4')] = 4  # 1.5*4 + 3*0.005=6.015 <=6.2
k_max[('6.2', 'H1')] = 2  # 2.2*2 + 1*0.005=4.405 <=6.2
k_max[('6.2', 'H2')] = 2  # 2.4*2 + 1*0.005=4.805 <=6.2
k_max[('6.2', 'H3')] = 2  # 2.3*2 + 1*0.005=4.605 <=6.2
k_max[('6.2', 'H4')] = 3  # 2.0*3 + 2*0.005=6.01 <=6.2

# 第三根原材料：7.8米
k_max[('7.8', 'W1')] = 4  # 1.6*4 + 3*0.005=6.415 <=7.8
k_max[('7.8', 'W2')] = 4  # 1.8*4 + 3*0.005=7.215 <=7.8
k_max[('7.8', 'W3')] = 4  # 1.7*4 + 3*0.005=6.815 <=7.8
k_max[('7.8', 'W4')] = 5  # 1.5*5 + 4*0.005=7.52 <=7.8
k_max[('7.8', 'H1')] = 3  # 2.2*3 + 2*0.005=6.61 <=7.8
k_max[('7.8', 'H2')] = 3  # 2.4*3 + 2*0.005=7.21 <=7.8
k_max[('7.8', 'H3')] = 3  # 2.3*3 + 2*0.005=6.91 <=7.8
k_max[('7.8', 'H4')] = 3  # 2.0*3 + 2*0.005=6.01 <=7.8

# ============== 建立整数线性规划模型 ==============
# 创建问题实例，命名为"Cutting_Optimization"，目标是最小化成本
prob = LpProblem("Cutting_Optimization", LpMinimize)

# 定义决策变量: z_{r,t} 表示使用原材料r切割段t的数量
raw_types = ['5.5', '6.2', '7.8']  # 原材料类型
seg_types = [s[0] for s in segments]  # 段类型
# 创建整数变量字典，变量名格式为"z_原材料类型_段类型"，下限为0，类型为整数
z = LpVariable.dicts("z", (raw_types, seg_types), lowBound=0, cat='Integer')

# 定义目标函数：最小化总成本
# 总成本 = sum(原材料单价 * 该原材料使用的总数量)
prob += lpSum([costs[i] * lpSum([z[raw_types[i]][t] for t in seg_types]) for i in range(len(raw_types))])

# 添加约束：每种段的生产量必须满足需求
for seg in segments:
    t, length, demand = seg  # 解构段信息：名称、长度、需求
    # 约束：sum(每种原材料切割该段的最大数量 * 使用该原材料的数量) >= 需求
    prob += lpSum([k_max[(r, t)] * z[r][t] for r in raw_types]) >= demand

# ============== 求解模型 ==============
prob.solve()  # 调用求解器求解

# ============== 输出结果 ==============
print("Status:", LpStatus[prob.status])  # 打印求解状态
print("Total Cost:", value(prob.objective))  # 打印最优目标值（最小总成本）

# ============== 详细使用情况计算 ==============
# 1. 计算每种原材料的使用数量
raw_usage = {r: sum(int(z[r][t].varValue) for t in seg_types) for r in raw_types}
# 计算总原材料成本
total_raw_cost = sum(raw_usage[r] * costs[i] for i, r in enumerate(raw_types))

# 2. 计算每种段的实际生产数量
segment_production = {t: sum(k_max[(r, t)] * int(z[r][t].varValue) for r in raw_types) for t in seg_types}

# 3. 计算总用料量和总段长度
# 总用料量 = sum(每种原材料使用数量 * 原材料长度)
total_material_length = sum(raw_usage[r] * lengths[i] for i, r in enumerate(raw_types))
# 总段长度 = sum(每种段的生产数量 * 段长度)
total_segment_length = sum(segment_production[t] * next(s[1] for s in segments if s[0] == t) for t in seg_types)

# 4. 计算余料和利用率
total_waste = total_material_length - total_segment_length  # 总余料
utilization = total_segment_length / total_material_length  # 原材料利用率

# 5. 输出详细切割方案
print("\n***** 详细切割方案 *****")
for r in raw_types:
    if raw_usage[r] > 0:
        print(f"\n原材料 {r}m (单价: {costs[raw_types.index(r)]}元) 使用 {raw_usage[r]} 根:")
        for t in seg_types:
            if z[r][t].varValue > 0:
                seg_length = next(s[1] for s in segments if s[0] == t)
                print(f"  - 切割 {int(z[r][t].varValue)} 根 × {k_max[(r, t)]} 段 {t} ({seg_length}m)")

# 输出生产统计信息
print("\n***** 生产统计 *****")
for t in seg_types:
    seg_length = next(s[1] for s in segments if s[0] == t)
    demand = next(s[2] for s in segments if s[0] == t)
    print(f"{t}段: 需求 {demand} 段 | 实际生产 {segment_production[t]} 段 | 长度 {seg_length}m")

# 输出资源利用统计
print("\n***** 资源利用统计 *****")
print(f"总用料量: {total_material_length} 米")
print(f"总段长度: {total_segment_length} 米")
print(f"总余料量: {total_waste:.2f} 米 (含锯口损失)")
print(f"原材料利用率: {utilization:.2%}")
print(f"总成本: {total_raw_cost} 元")

# 6. 计算利润（假设窗框单价已知）
order_profit = {
    'W1': 480, 'H1': 480,  # 学校教学楼
    'W2': 680, 'H2': 680,  # 酒店客房
    'W3': 550, 'H3': 550,  # 医院病房
    'W4': 420, 'H4': 420   # 政府办公楼
}
# 计算总收入（注意：这里假设每对W和H段组成一个窗户）
total_revenue = sum(
    min(segment_production[t], next(s[2] for s in segments if s[0] == t)) * order_profit[t] / 2 
    for t in seg_types
)
print(f"\n总收益: {total_revenue:.2f} 元")
print(f"总利润: {total_revenue - total_raw_cost:.2f} 元")