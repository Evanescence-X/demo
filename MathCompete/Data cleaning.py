import numpy as np
import pandas as pd
from scipy import stats

# 导入表
file_path = r"C:\Users\17813\Desktop\数模\B\附件.xlsx"
sheet_name = "Sheet1"
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 基本属性的检查
print("数据原始形状:",df.shape)
print("前十行数据:",df.head(10))
print("表的描述性统计:",df.describe())

# 处理空值
if df.isnull().any().any():
    df = df.dropna()

# 处理重复值
if df.duplicated().any().any():
    df = df.drop(df.duplicated())

# 定义 IQR 和 Z-score 方法
def detect_outliers(df, column, method='iqr', threshold=1.5):
    data = df[column]

    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = df[(data < lower_bound) | (data > upper_bound)]

    else:
        z_scores = np.abs(stats.zscore(data))
        outliers = df[z_scores > threshold]

    return outliers.index, len(outliers)

numeric_cols = ['原材料长度 (米)', '缺陷位置 (米)', '缺陷长度 (米)', '单价（元/根）']
clean_df = df.copy()


for col in numeric_cols:
    print(f"\n=== 正在处理列: {col} ===")

    # 方法1: IQR检测
    outlier_idx, outlier_count = detect_outliers(df, col, method='iqr', threshold=1.5)
    print(f"IQR方法检测到异常值数量: {outlier_count}")
    if outlier_count > 0:
        print("异常值数据:")
        print(df.loc[outlier_idx, col])

    # 方法2: Z-score检测
    outlier_idx, outlier_count = detect_outliers(df, col, method='zscore', threshold=3)
    print(f"Z-score方法检测到异常值数量: {outlier_count}")
    if outlier_count > 0:
        print("异常值数据:")
        print(df.loc[outlier_idx, col])

    # 业务逻辑验证 - 以缺陷位置为例
    if col == '缺陷位置 (米)':
        # 检查缺陷位置是否超过原材料长度
        invalid_pos = df[df['缺陷位置 (米)'] + df['缺陷长度 (米)'] > df['原材料长度 (米)']]
        print(f"\n缺陷位置超过原材料长度的记录数: {len(invalid_pos)}")
        if len(invalid_pos) > 0:
            print("无效记录:")
            print(invalid_pos)
            # 从清洗后的数据中移除这些记录
            clean_df = clean_df.drop(invalid_pos.index)

    # 处理异常值 - 这里选择用中位数替换IQR检测出的异常值
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    median_val = df[col].median()
    clean_df.loc[(clean_df[col] < lower_bound) | (clean_df[col] > upper_bound), col] = median_val


# 保存清洗后的数据
clean_file_path = '清洗后_附件.xlsx'
clean_df.to_excel(clean_file_path, index=False)
print(f"\n数据清洗完成，结果已保存到: {clean_file_path}")

# 显示清洗后的数据信息
print("\n清洗后数据形状:", clean_df.shape)
print("\n清洗后数据描述性统计:")
print(clean_df.describe())

# 验证清洗结果
print("\n清洗后各列异常值数量(IQR方法):")
for col in numeric_cols:
    _, outlier_count = detect_outliers(clean_df, col, method='iqr')
    print(f"{col}: {outlier_count}")