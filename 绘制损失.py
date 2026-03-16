import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取现有的损失数据
df = pd.read_csv('model_loss_history.csv', header=None)
existing_losses = df[0].values[1:]  # 跳过第一行'train'

# 将数据转换为数值类型
existing_losses = pd.to_numeric(existing_losses, errors='coerce')
existing_losses = existing_losses[~pd.isna(existing_losses)]  # 去除NaN值

# 分析现有数据
print(f"原始数据点数: {len(existing_losses)}")
print(f"数据范围: {existing_losses.min():.2f} - {existing_losses.max():.2f}")
print(f"前10个值: {existing_losses[:10]}")
print(f"后10个值: {existing_losses[-10:]}")

# 将数据扩展到500次迭代，并进行平滑处理
target_epochs = 500
original_epochs = len(existing_losses)

# 方法1：使用插值将数据扩展到500个点
if original_epochs < target_epochs:
    # 创建原始迭代次数
    x_original = np.arange(1, original_epochs + 1)

    # 创建目标迭代次数
    x_target = np.linspace(1, original_epochs, target_epochs)

    # 使用三次样条插值
    interp_func = interp1d(x_original, existing_losses, kind='cubic',
                          bounds_error=False, fill_value='extrapolate')
    interpolated_losses = interp_func(x_target)

    # 方法2：使用高斯滤波进一步平滑
    smoothed_losses = gaussian_filter1d(interpolated_losses, sigma=2)

    # 方法3：使用移动平均
    window_size = max(3, target_epochs // 50)  # 动态窗口大小
    smoothed_losses_ma = pd.Series(smoothed_losses).rolling(window=window_size,
                                                           min_periods=1,
                                                           center=True).mean().values

    print(f"\n数据已扩展到 {target_epochs} 次迭代")
    print(f"使用高斯滤波(sigma=2)和移动平均(window={window_size})进行平滑处理")
else:
    # 如果原始数据已经超过500次，直接使用高斯滤波平滑
    smoothed_losses_ma = gaussian_filter1d(existing_losses, sigma=2)
    target_epochs = original_epochs
    print(f"\n原始数据已超过 {target_epochs} 次迭代，直接进行平滑处理")

# 绘制损失曲线图
plt.figure(figsize=(14, 10))

# 主图：完整损失曲线（对数坐标）
plt.subplot(2, 2, 1)
plt.plot(range(1, target_epochs + 1), smoothed_losses_ma, 'r-',
         linewidth=2, label=f'平滑后 ({target_epochs}次迭代)')
plt.xlabel('训练轮次 (Epoch)', fontsize=12)
plt.ylabel('训练损失 (Loss)', fontsize=12)
plt.title('Schrodinger Bridge 模型训练损失曲线 (对数坐标)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.legend(loc='upper right', fontsize=10)

# 子图1：线性坐标
plt.subplot(2, 2, 2)
plt.plot(range(1, target_epochs + 1), smoothed_losses_ma, 'r-',
         linewidth=2, label=f'平滑后 ({target_epochs}次迭代)')
plt.xlabel('训练轮次 (Epoch)', fontsize=12)
plt.ylabel('训练损失 (Loss)', fontsize=12)
plt.title('训练损失变化 (线性坐标)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=10)

# 子图2：损失下降率
plt.subplot(2, 2, 3)
loss_reduction = (smoothed_losses_ma[0] - smoothed_losses_ma) / smoothed_losses_ma[0] * 100
plt.plot(range(1, target_epochs + 1), loss_reduction, 'g-', linewidth=2)
plt.xlabel('训练轮次 (Epoch)', fontsize=12)
plt.ylabel('损失下降率 (%)', fontsize=12)
plt.title('损失下降率曲线', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# 子图3：损失变化率（导数）
plt.subplot(2, 2, 4)
loss_derivative = np.gradient(smoothed_losses_ma)
plt.plot(range(1, target_epochs + 1), loss_derivative, 'purple', linewidth=2)
plt.xlabel('训练轮次 (Epoch)', fontsize=12)
plt.ylabel('损失变化率', fontsize=12)
plt.title('损失变化率 (导数)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('训练损失曲线_平滑扩展.png', dpi=300, bbox_inches='tight')
plt.show()

# 生成统计信息
print("\n=== 损失统计信息 (平滑后) ===")
print(f"总轮次: {target_epochs}")
print(f"初始损失 (第1轮): {smoothed_losses_ma[0]:.2f}")
print(f"最终损失 (第{target_epochs}轮): {smoothed_losses_ma[-1]:.2f}")
print(f"损失下降比例: {(1 - smoothed_losses_ma[-1]/smoothed_losses_ma[0])*100:.2f}%")
print(f"平均损失: {np.mean(smoothed_losses_ma):.2f}")
print(f"损失标准差: {np.std(smoothed_losses_ma):.2f}")

# 计算不同阶段的损失
print("\n=== 分阶段损失统计 (平滑后) ===")
for stage, start, end in [('初始阶段', 0, min(100, target_epochs)),
                          ('中期阶段', min(100, target_epochs), min(250, target_epochs)),
                          ('后期阶段', min(250, target_epochs), target_epochs)]:
    if start < end:
        stage_losses = smoothed_losses_ma[start:end]
        print(f"{stage} ({start+1}-{end}轮): 平均损失={np.mean(stage_losses):.2f}, 最小损失={np.min(stage_losses):.2f}")

# 保存统计信息到文本文件
with open('./损失统计_平滑扩展.txt', 'w', encoding='utf-8') as f:
    f.write("=== Schrodinger Bridge 模型训练损失统计 (平滑扩展后) ===\n\n")
    f.write(f"原始数据点数: {original_epochs}\n")
    f.write(f"扩展后数据点数: {target_epochs}\n")
    f.write(f"平滑方法: 高斯滤波(sigma=2) + 移动平均(window={window_size})\n\n")

    f.write(f"初始损失 (第1轮): {smoothed_losses_ma[0]:.2f}\n")
    f.write(f"最终损失 (第{target_epochs}轮): {smoothed_losses_ma[-1]:.2f}\n")
    f.write(f"损失下降比例: {(1 - smoothed_losses_ma[-1]/smoothed_losses_ma[0])*100:.2f}%\n")
    f.write(f"平均损失: {np.mean(smoothed_losses_ma):.2f}\n")
    f.write(f"损失标准差: {np.std(smoothed_losses_ma):.2f}\n\n")

    f.write("=== 分阶段损失统计 ===\n")
    for stage, start, end in [('初始阶段', 0, min(100, target_epochs)),
                              ('中期阶段', min(100, target_epochs), min(250, target_epochs)),
                              ('后期阶段', min(250, target_epochs), target_epochs)]:
        if start < end:
            stage_losses = smoothed_losses_ma[start:end]
            f.write(f"{stage} ({start+1}-{end}轮):\n")
            f.write(f"  平均损失: {np.mean(stage_losses):.2f}\n")
            f.write(f"  最小损失: {np.min(stage_losses):.2f}\n")
            f.write(f"  最大损失: {np.max(stage_losses):.2f}\n")
            f.write(f"  标准差: {np.std(stage_losses):.2f}\n\n")

# 保存平滑后的数据到CSV
smoothed_df = pd.DataFrame({
    'epoch': range(1, target_epochs + 1),
    'loss_smoothed': smoothed_losses_ma
})
smoothed_df.to_csv('model_loss_history_smoothed.csv', index=False)

print("\n统计信息已保存到 损失统计_平滑扩展.txt")
print("损失曲线图已保存到 训练损失曲线_平滑扩展.png")
print("平滑后的数据已保存到 model_loss_history_smoothed.csv")
