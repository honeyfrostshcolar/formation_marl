import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. 路径配置
# =========================================================
CSV_DIR = Path("/home/nankai/formation_test/paperdata/full_star_uniform[1,3]_latest/tb_csv")          # 改成你的 CSV 文件夹
OUT_DIR = Path("/home/nankai/formation_test/paperdata/full_star_uniform[1,3]_latest/csv_plot")   # 输出图文件夹
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train_reward": "run-.-tag-Eval_AvgReward.csv",
    "eval_reward": "run-.-tag-Eval_star_map_Reward.csv",
    "success": "run-.-tag-Eval_star_map_Success.csv",
    "collision": "run-.-tag-Eval_star_map_Collision.csv",
    "formation_error": "run-.-tag-Eval_star_map_FormationError.csv",
    "actor_loss": "run-.-tag-Loss_Actor_Loss.csv",
    "critic_loss": "run-.-tag-Loss_Critic_Loss.csv",
}

# =========================================================
# 2. 全局风格设置（更适合论文）
# =========================================================
def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


# =========================================================
# 3. 读取 TensorBoard CSV
#    兼容一般格式：Wall time / Step / Value
# =========================================================
def read_tb_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    col_map = {c.lower().strip(): c for c in df.columns}

    if "step" not in col_map or "value" not in col_map:
        raise ValueError(
            f"{path.name} does not contain 'Step' and 'Value' columns. "
            f"Columns found: {df.columns.tolist()}"
        )

    out = pd.DataFrame({
        "step": pd.to_numeric(df[col_map["step"]], errors="coerce"),
        "value": pd.to_numeric(df[col_map["value"]], errors="coerce"),
    }).dropna()

    out = out.sort_values("step").reset_index(drop=True)
    return out


# =========================================================
# 4. 平滑函数
#    用 rolling mean，只用于画图
# =========================================================
def smooth_series(y, window=9):
    y = np.asarray(y, dtype=float)
    if len(y) < 3 or window <= 1:
        return y

    window = min(window, len(y))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return y

    return pd.Series(y).rolling(
        window=window, center=True, min_periods=1
    ).mean().to_numpy()


# =========================================================
# 5. 单子图绘制函数
# =========================================================
def plot_metric(
    ax,
    csv_path,
    panel_title,
    ylabel,
    smooth_window=9,
    raw_color="#B7C9D9",       # 浅蓝灰
    smooth_color="#E86E00",    # 主色：橙色
    ylim=None,
    show_xlabel=False
):
    df = read_tb_csv(csv_path)
    x = df["step"].to_numpy()
    y = df["value"].to_numpy()
    y_s = smooth_series(y, smooth_window)

    # raw: 浅色细线
    ax.plot(x, y, color=raw_color, linewidth=1.0, alpha=0.95, zorder=1)

    # smoothed: 深色粗线
    ax.plot(x, y_s, color=smooth_color, linewidth=1.8, zorder=2)

    ax.set_title(panel_title, pad=4)
    ax.set_ylabel(ylabel)

    if show_xlabel:
        ax.set_xlabel("Episode")
    else:
        ax.set_xlabel("")

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

    # 收紧边框
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # 左右留一点边距，避免最左最右太挤
    if len(x) > 1:
        x_margin = 0.02 * (x.max() - x.min())
        ax.set_xlim(x.min() - x_margin, x.max() + x_margin)


# =========================================================
# 6. 正文主图：2x2 convergence figure
#    推荐放正文
# =========================================================
def make_main_convergence_figure():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))

    # (a) Eval reward
    plot_metric(
        axes[0, 0],
        CSV_DIR / FILES["eval_reward"],
        panel_title="(a) Eval. Reward",
        ylabel="Avg. Reward",
        smooth_window=3,
        show_xlabel=False,
    )

    # (b) Success
    plot_metric(
        axes[0, 1],
        CSV_DIR / FILES["success"],
        panel_title="(b) Success",
        ylabel="Success Rate",
        smooth_window=7,
        ylim=(-0.02, 1.05),
        show_xlabel=False,
    )

    # (c) Collision
    plot_metric(
        axes[1, 0],
        CSV_DIR / FILES["collision"],
        panel_title="(c) Collision",
        ylabel="Collision Rate",
        smooth_window=7,
        ylim=(-0.02, 1.05),
        show_xlabel=True,
    )

    # (d) Formation error
    plot_metric(
        axes[1, 1],
        CSV_DIR / FILES["formation_error"],
        panel_title="(d) Formation Error",
        ylabel="Error (m)",
        smooth_window=3,
        show_xlabel=True,
    )

    # 统一布局
    plt.tight_layout(pad=1.0, w_pad=1.2, h_pad=1.2)

    # 输出多种格式
    fig.savefig(OUT_DIR / "fig_convergence_main.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_convergence_main.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_convergence_main.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# =========================================================
# 7. 附录/补充图：训练与优化诊断
#    不建议放正文主图
# =========================================================
def make_diagnostics_figure():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))

    # Training reward
    plot_metric(
        axes[0],
        CSV_DIR / FILES["train_reward"],
        panel_title="(a) Training Reward",
        ylabel="Avg. Step Reward",
        smooth_window=3,
        show_xlabel=True,
        raw_color="#C7D3DD",
        smooth_color="#1F77B4",
    )

    # Actor loss
    plot_metric(
        axes[1],
        CSV_DIR / FILES["actor_loss"],
        panel_title="(b) Actor Loss",
        ylabel="Loss",
        smooth_window=21,
        show_xlabel=True,
        raw_color="#D9D9D9",
        smooth_color="#2CA02C",
    )

    # Critic loss
    plot_metric(
        axes[2],
        CSV_DIR / FILES["critic_loss"],
        panel_title="(c) Critic Loss",
        ylabel="Loss",
        smooth_window=21,
        show_xlabel=True,
        raw_color="#D9D9D9",
        smooth_color="#D62728",
    )

    plt.tight_layout(pad=1.0, w_pad=1.2)
    fig.savefig(OUT_DIR / "fig_diagnostics_appendix.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_diagnostics_appendix.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_diagnostics_appendix.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# =========================================================
# 8. 如你还想保留单张图，也一起输出
#    方便你自己看
# =========================================================
def make_single_figure(csv_file, title, ylabel, out_name,
                       smooth_window=9, ylim=None,
                       raw_color="#B7C9D9", smooth_color="#E86E00"):
    fig, ax = plt.subplots(figsize=(3.25, 2.45))
    plot_metric(
        ax=ax,
        csv_path=CSV_DIR / csv_file,
        panel_title=title,
        ylabel=ylabel,
        smooth_window=smooth_window,
        raw_color=raw_color,
        smooth_color=smooth_color,
        ylim=ylim,
        show_xlabel=True
    )
    plt.tight_layout(pad=0.8)
    fig.savefig(OUT_DIR / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_name}.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_all_single_figures():
    make_single_figure(
        FILES["eval_reward"], "Eval. Reward", "Avg. Reward",
        "single_eval_reward", smooth_window=9
    )
    make_single_figure(
        FILES["success"], "Success", "Success Rate",
        "single_success", smooth_window=7, ylim=(-0.02, 1.05)
    )
    make_single_figure(
        FILES["collision"], "Collision", "Collision Rate",
        "single_collision", smooth_window=7, ylim=(-0.02, 1.05)
    )
    make_single_figure(
        FILES["formation_error"], "Formation Error", "Error (m)",
        "single_formation_error", smooth_window=9
    )
    make_single_figure(
        FILES["train_reward"], "Training Reward", "Avg. Step Reward",
        "single_training_reward", smooth_window=9, smooth_color="#1F77B4"
    )
    make_single_figure(
        FILES["actor_loss"], "Actor Loss", "Loss",
        "single_actor_loss", smooth_window=21, raw_color="#D9D9D9", smooth_color="#2CA02C"
    )
    make_single_figure(
        FILES["critic_loss"], "Critic Loss", "Loss",
        "single_critic_loss", smooth_window=21, raw_color="#D9D9D9", smooth_color="#D62728"
    )


# =========================================================
# 9. main
# =========================================================
def main():
    setup_matplotlib()

    # 正文主图
    make_main_convergence_figure()

    # 附录图
    make_diagnostics_figure()

    # 可选：导出所有单张图
    make_all_single_figures()

    print(f"All figures have been saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()