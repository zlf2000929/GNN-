import pandas as pd

# 加载Excel文件
file_path = '111.xlsx'
df = pd.read_excel(file_path)

# 检查数据
print("原始数据:")
print(df.head())

# 获取第二列到最后一列的列名
columns = df.columns[1:]  # 从第二列开始

# 对这些列进行处理
for col in columns:
    # 将小于0.5的值替换为1-x
    df[col] = df[col].apply(lambda x: 1-x if x < 0.5 else x)

# 查看处理后的数据
print("处理后的数据:")
print(df.head())

# 保存到新的Excel文件
output_path = 'output_111.xlsx'
df.to_excel(output_path, index=False)

print(f"文件已保存为 {output_path}")