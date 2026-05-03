import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 全局学术级画图参数设置
# ==========================================
plt.rcParams.update({
    "font.family": "serif",        
    "font.size": 12,
    "axes.labelsize": 13,          
    "axes.titlesize": 14,          
    "axes.titleweight": "bold",   
    "xtick.labelsize": 10,         # 5个模型标签较多，字号稍微缩小
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,             
    "savefig.bbox": "tight",       
})

def plot_comprehensive_summary():
    # ==========================================
    # 2. 数据填入区 (替换为你真实跑出来的均值和标准差)
    # ==========================================
    # 按照: Full(None), Full(Fixed-2), Full(Uni), No-Belief, No-Comm 的顺序
    labels = ['Full\n(None)', 'Full\n(Fixed-2)', 'Full\n(Uni[1,3])', 'No-Belief\n(Fixed-2)', 'No-Comm\n(Fixed-2)']
    
    data = {
        'Success Rate': {
            'mean': [1.000, 0.98, 0.98, 0.96, 0.98],
            'std':  [0.000, 0.14, 0.14, 0.196, 0.14]
        },
        'Formation Error': {
            'mean': [0.476, 0.504, 0.431, 0.561, 0.497],
            'std':  [0.167, 0.279, 0.228, 0.353, 0.211]
        },
        'Collision Rate': {
            'mean': [0.00, 0.02, 0.02, 0.04, 0.02],
            'std':  [0.00, 0.14, 0.14, 0.196, 0.14]
        },
        'Avg. Reward': {
            'mean': [33.523, 13.692, 33.148, -10.627, 37.947],
            'std':  [56.501, 75.976, 62.229, 82.358, 71.743]
        }
    }

    # 配色方案：前三个Full模型用蓝色系(渐变)，消融模型用警告色(橙、红)
    colors = ['#2878B5', '#60A0D0', '#9AC9DB', '#F8AC8C', '#C82423']
    
    # ==========================================
    # 3. 绘图逻辑
    # ==========================================
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    axs = axs.flatten()
    
    metrics = ['Success Rate', 'Formation Error', 'Collision Rate', 'Avg. Reward']
    y_labels = ['Success Rate', 'Formation Error (m)', 'Collision Rate', 'Average Reward']
    
    x_pos = np.arange(len(labels))
    bar_width = 0.6
    
    for i, metric in enumerate(metrics):
        ax = axs[i]
        means = data[metric]['mean']
        stds = data[metric]['std']
        
        bars = ax.bar(x_pos, means, bar_width, 
                      yerr=stds, 
                      color=colors, 
                      alpha=0.9, 
                      edgecolor='black',
                      linewidth=1.2,
                      capsize=5,
                      error_kw={'elinewidth': 1.5, 'markeredgewidth': 1.5})
        
        ax.set_title(metric)
        ax.set_ylabel(y_labels[i])
        ax.set_xticks(x_pos)
        # 针对拥挤的标签，倾斜15度
        ax.set_xticklabels(labels, rotation=15, ha='center')
        
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_axisbelow(True) 
        
        # 优化坐标轴范围
        if metric == 'Success Rate':
            ax.set_ylim(0, 1.2)
        elif metric == 'Collision Rate':
            ax.set_ylim(0, max(means) + max(stds) + 0.1)
            
        # 在柱子上标注具体数值 (针对窄柱体，数值略微旋转或调整字体)
        for bar in bars:
            yval = bar.get_height()
            text_val = f'{yval:.2f}' if abs(yval) < 10 else f'{yval:.1f}'
            offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            # 如果是负数（如Reward），文本写在柱子下方
            if yval < 0:
                ax.text(bar.get_x() + bar.get_width()/2, yval - offset, 
                        text_val, ha='center', va='top', fontsize=9, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, yval + offset, 
                        text_val, ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    
    # ==========================================
    # 4. 保存输出
    # ==========================================
    save_dir = "/home/nankai/formation_test/paperdata"
    os.makedirs(save_dir, exist_ok=True)
    
    pdf_path = os.path.join(save_dir, 'fig_comprehensive_summary.pdf')
    png_path = os.path.join(save_dir, 'fig_comprehensive_summary.png')
    
    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png', dpi=300)
    
    print(f"✅ 综合汇总图表已生成:\n  - {pdf_path}\n  - {png_path}")
    plt.close(fig)

if __name__ == '__main__':
    plot_comprehensive_summary()