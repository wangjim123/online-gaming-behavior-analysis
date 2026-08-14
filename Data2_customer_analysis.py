import matplotlib.pyplot as plt
import numpy as np
import os

def main(df1):
    df = df1.copy()
    # 1. 資料清洗與分群統計
    payer_df = df[
        df["InAppPurchaseAmount"].notnull() & (df["InAppPurchaseAmount"] > 0)
        ].copy()

    segment_summary = (
        payer_df.groupby("SpendingSegment")
        .agg(
            player_count=("UserID", "count"),
            total_revenue=("InAppPurchaseAmount", "sum"),
            avg_revenue=("InAppPurchaseAmount", "mean"),
        )
        .reindex(["Minnow", "Dolphin", "Whale"])
    )

    segment_summary["player_share"] = (
            segment_summary["player_count"]
            / segment_summary["player_count"].sum()
            * 100
    )
    segment_summary["revenue_share"] = (
            segment_summary["total_revenue"]
            / segment_summary["total_revenue"].sum()
            * 100
    )

    # 2. 建立 1 列 2 欄圖表 (圖 1 與 圖 2)
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)

    # --- 圖 1：玩家人數佔比 vs. 營收佔比 (Pareto 80/20) ---
    x = np.arange(len(segment_summary.index))
    width = 0.35

    rects1 = ax1.bar(
        x - width / 2,
        segment_summary["player_share"],
        width,
        label="User Share (%)",
        color="#3498db",
        edgecolor="black",
        alpha=0.85,
    )
    rects2 = ax1.bar(
        x + width / 2,
        segment_summary["revenue_share"],
        width,
        label="Revenue Share (%)",
        color="#e74c3c",
        edgecolor="black",
        alpha=0.85,
    )

    ax1.set_title(
        "1. User Share vs Revenue Share (Pareto 80/20)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Minnow", "Dolphin", "Whale"], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Percentage (%)", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right", frameon=True)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#c0392b",
        )

    # --- 圖 2：各分群平均單人消費 (ARPU) ---
    colors_arpu = ["#95a5a6", "#f39c12", "#9b59b6"]
    bars2 = ax2.bar(
        range(len(segment_summary)),
        segment_summary["avg_revenue"],
        color=colors_arpu,
        edgecolor="black",
        width=0.45,
        alpha=0.85,
    )
    ax2.set_title(
        "2. Average Spend per User (ARPU)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax2.set_xticks(range(len(segment_summary)))
    ax2.set_xticklabels(["Minnow", "Dolphin", "Whale"], fontsize=11, fontweight="bold")
    ax2.set_ylabel("Average Spend ($ USD)", fontsize=11)
    ax2.set_ylim(0, segment_summary["avg_revenue"].max() * 1.15)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(
            f"${h:,.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )

    plt.tight_layout()
    # 存檔
    os.makedirs("Data2_png", exist_ok=True)
    output_path = os.path.join("Data2_png", "Data2_customer_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #---------------------------------------------------------
    plt.show()