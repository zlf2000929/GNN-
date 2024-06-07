import os
import shutil

source_folder = r"Benchmark dataset"
target_folder = r"Benchmark dataset_pdb"

os.makedirs(target_folder, exist_ok=True)

# 遍历源文件夹中的所有子文件夹
for subdir in os.listdir(source_folder):
    # 检查子文件夹名是否符合'{x}pos_{y}_{z}'的格式
    if "pos" in subdir:
        parts = subdir.split('_')
        if len(parts) == 3 and parts[0].endswith("pos"):
            x = parts[0][:-3]  # 提取x部分，去除'pos'
            y = parts[1]       # 提取y部分
            subdir_path = os.path.join(source_folder, subdir)

            # 遍历子文件夹中的所有文件
            for file in os.listdir(subdir_path):
                # 检查文件扩展名是否为'.pdb'
                if file.endswith(".pdb"):
                    # 构造新的文件名和路径
                    new_filename = f"{x}_{y}.pdb"
                    new_filepath = os.path.join(target_folder, new_filename)
                    file_path = os.path.join(subdir_path, file)
                    # 复制并重命名文件到目标文件夹
                    shutil.copy(file_path, new_filepath)

print("文件提取和重命名完成。")