import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 全局学术级画图参数设置
# ==========================================
plt.rcParams.update({
    "font.family": "serif",        # 学术论文常用 serif 字体
    "font.size": 12,
    "axes.labelsize": 13,          # 坐标轴标签字号
    "axes.titlesize": 14,          # 子图标题字号
    "axes.titleweight": "bold",   # 标题加粗
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,             # 高分辨率
    "savefig.bbox": "tight",       # 保存时去除白边
})

def plot_figure_5():
    # ==========================================
    # 2. 数据填入区 (替换为你真实跑出来的 mean 和 std)
    # ==========================================
    labels = ['Full + none\n(No Delay)', 'Full + fixed-2\n(Constant)', 'Full + uniform[1,3]\n(Random)']
    
    # 格式: [none的数值, fixed-2的数值, uniform[1,3]的数值]
    data = {
        'Success Rate': {
            'mean': [1.000, 0.98, 0.98],
            'std':  [0.000, 0.14, 0.14]
        },
        'Formation Error': {
            'mean': [0.476, 0.504, 0.431],
            'std':  [0.167, 0.279, 0.228]
        },
        'Collision Rate': {
            'mean': [0.00, 0.02, 0.02],
            'std':  [0.00, 0.14, 0.14]
        },
        'Avg. Reward': {
            'mean': [33.523, 13.692, 33.148],
            'std':  [56.501, 75.976, 62.229]
        }
    }

    # 配色方案：学术期刊常用的高对比度色盲友好配色
    colors = ['#2878B5', '#9AC9DB', '#F8AC8C']
    
    # ==========================================
    # 3. 绘图逻辑
    # ==========================================
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    
    metrics = ['Success Rate', 'Formation Error', 'Collision Rate', 'Avg. Reward']
    y_labels = ['Success Rate', 'Formation Error (m)', 'Collision Rate', 'Average Reward']
    
    x_pos = np.arange(len(labels))
    bar_width = 0.5
    
    for i, metric in enumerate(metrics):
        ax = axs[i]
        means = data[metric]['mean']
        stds = data[metric]['std']
        
        # 绘制带误差棒的柱状图
        bars = ax.bar(x_pos, means, bar_width, 
                      yerr=stds, 
                      color=colors, 
                      alpha=0.9, 
                      edgecolor='black',
                      linewidth=1.2,
                      capsize=6,        # 误差棒的横线宽度
                      error_kw={'elinewidth': 1.5, 'markeredgewidth': 1.5})
        
        # 设置标题和标签
        ax.set_title(metric)
        ax.set_ylabel(y_labels[i])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        
        # 添加背景网格线 (仅Y轴)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_axisbelow(True) 
        
        # 优化坐标轴范围
        if metric == 'Success Rate':
            ax.set_ylim(0, 1.19) # 留出顶部空间写数字
        elif metric == 'Collision Rate':
            ax.set_ylim(0, max(means) + max(stds) + 0.05)
            
        # 在柱子上标注具体数值
        for bar in bars:
            yval = bar.get_height()
            text_val = f'{yval:.2f}' if yval < 10 else f'{yval:.1f}'
            offset = 0.02 * ax.get_ylim()[1]
            ax.text(bar.get_x() + bar.get_width()/2, yval + offset, 
                    text_val, ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    
    # ==========================================
    # 4. 保存输出
    # ==========================================
    save_dir = "/home/nankai/formation_test/paperdata"
    os.makedirs(save_dir, exist_ok=True)
    
    pdf_path = os.path.join(save_dir, 'fig5_comm_robustness.pdf')
    png_path = os.path.join(save_dir, 'fig5_comm_robustness.png')
    
    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png', dpi=300)
    
    print(f"✅ Figure 5 已保存至:\n  - {pdf_path}\n  - {png_path}")
    plt.close(fig)

if __name__ == '__main__':
    plot_figure_5()