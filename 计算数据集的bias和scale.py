# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from tqdm import tqdm
# !!!!!!!!!!!!!!!!!!!!!
# 用于计算 schrodinger_bridge_model.yml 中的 bias 和 scale
# 该脚本会扫描指定目录下的所有 .npz 文件，计算每个变量的均值和标准差
# 并输出适用于 yml 文件的格式
# !!!!!!!!!!!!!!!!!!!!!
# ==============================================================================
# ---                           CONFIGURATION                          ---
# ==============================================================================

# TODO: 请将此路径修改为您在Linux服务器上存放 .npz 文件的目录
NPZ_DIR = "/public3/home/scg4074/tzq/schrodinger-bridge-sr-micrometeorology/prepare_npz"

# 我们需要计算统计量的变量列表
# 这应该与您 .yml 文件中的 input_variable_names 和 target_variable_names 的总和相对应
# VARIABLES_TO_COMPUTE = [
#     'hr_tm002m', 'lr_tm002m', 'z', 'lu', 'tsk', 'swdown', 'glw', 'hfx', 
#     'lh', 'psfc', 'pblh', 'u10', 'v10', 't_900', 'u_900', 'v_900', 'q_900', 
#     't_850', 'u_850', 'v_850', 'q_850'
# ]

VARIABLES_TO_COMPUTE = [
    'hr_tm002m', 'lr_tm002m', 'z', 'lu', 'tsk', 'swdown', 'glw', 'hfx',
    'lh', 'psfc', 'pblh', 'u10', 'v10',
    't_ml2', 'u_ml2', 'v_ml2', 'q_ml2',
    't_ml5', 'u_ml5', 'v_ml5', 'q_ml5'
]
# ==============================================================================
# ---                       COMPUTATION LOGIC                          ---
# ==============================================================================

def compute_statistics():
    """
    使用 Welford's algorithm 在线计算大量数据的均值和标准差，以避免内存溢出。
    """
    all_files = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
    if not all_files:
        print(f"错误: 在目录 '{NPZ_DIR}' 中未找到 .npz 文件。")
        return

    print(f"找到 {len(all_files)} 个 .npz 文件，开始计算统计量...")

    # 为 Welford's algorithm 初始化累加器
    stats = {
        var: {'count': 0, 'mean': 0.0, 'm2': 0.0}
        for var in VARIABLES_TO_COMPUTE
    }

    # 遍历所有文件
    for file_path in tqdm(all_files, desc="正在扫描文件"):
        try:
            with np.load(file_path) as data:
                for var_name in VARIABLES_TO_COMPUTE:
                    if var_name in data:
                        # 获取数据并展平为一维
                        array_data = data[var_name].flatten().astype(np.float64)
                        
                        # 更新统计量
                        s = stats[var_name]
                        for x in array_data:
                            s['count'] += 1
                            delta = x - s['mean']
                            s['mean'] += delta / s['count']
                            delta2 = x - s['mean']
                            s['m2'] += delta * delta2
                    else:
                        print(f"\n警告: 在文件 {os.path.basename(file_path)} 中找不到变量 '{var_name}'，跳过。")
        except Exception as e:
            print(f"\n处理文件 {os.path.basename(file_path)} 时出错: {e}")

    print("\n统计量计算完成。正在生成YAML配置...")

    # 计算最终的均值和标准差
    final_stats = {}
    for var_name, s in stats.items():
        if s['count'] < 2:
            final_stats[var_name] = {'mean': 0.0, 'std': 1.0}
        else:
            mean = s['mean']
            std = np.sqrt(s['m2'] / (s['count'] - 1))
            # 如果标准差接近于0（例如对于像LANDMASK这样的恒定变量），则将其设置为1.0以避免除以零
            if std < 1e-6:
                print(f"警告: 变量 '{var_name}' 的标准差非常小 ({std:.2e})，将 scale 设置为 1.0。")
                std = 1.0
            final_stats[var_name] = {'mean': mean, 'std': std}

    # 打印YAML格式的输出
    print("\n" + "="*50)
    print("请将以下内容复制并替换到您的 schrodinger_bridge_model.yml 文件中的 'data' 部分：")
    print("="*50 + "\n")

    print("  biases:")
    for var_name in VARIABLES_TO_COMPUTE:
        # 在 .yml 文件中，我们使用截断后的键名
        key = var_name[:5]
        # 对于特殊的，例如 t_900，键名就是它本身
        if '_' in var_name:
            key = var_name
        # 特殊处理 lr_tm002m -> lr_tm
        if var_name == 'lr_tm002m':
            key = 'lr_tm'
        if var_name == 'hr_tm002m':
            key = 'hr_tm'
            
        print(f"    {key}: {final_stats[var_name]['mean']:.6f}")

    print("\n  scales:")
    for var_name in VARIABLES_TO_COMPUTE:
        key = var_name[:5]
        if '_' in var_name:
            key = var_name
        if var_name == 'lr_tm002m':
            key = 'lr_tm'
        if var_name == 'hr_tm002m':
            key = 'hr_tm'
            
        print(f"    {key}: {final_stats[var_name]['std']:.6f}")

    print("\n" + "="*50)
    print("复制完成后，请确保您的 .yml 文件格式正确。")
    print("="*50)


if __name__ == "__main__":
    compute_statistics()
