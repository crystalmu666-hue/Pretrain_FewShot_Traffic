import pandas as pd
import os

# 1. 设置路径
raw_dir = "data/raw"
output_path = os.path.join(raw_dir, "  .csv")

# 2. 获取目录下所有的 CSV 文件
# 关键过滤：文件名必须包含 'pcap' 或 'ISCX' (这是 CICIDS 的特征)，且不能包含 'UNSW'
all_files = [
    f for f in os.listdir(raw_dir)
    if f.endswith('.csv') and 'UNSW' not in f and 'combined' not in f
]

print(f"📡 检测到待合并的 CICIDS 文件: {all_files}")

# 3. 开始合并
combined_list = []
for file in all_files:
    file_path = os.path.join(raw_dir, file)
    print(f"正在读取: {file} ...")

    # 同样建议 nrows 限制，防止内存溢出
    df = pd.read_csv(file_path, low_memory=False, nrows=80000)
    combined_list.append(df)

# 4. 执行合并与打乱
final_df = pd.concat(combined_list, axis=0, ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. 保存
final_df.to_csv(output_path, index=False)

print(f"\n✅ 合并完成！")
print(f"📊 最终类别分布:\n{final_df[' Label'].value_counts() if ' Label' in final_df.columns else '未找到标签列'}")