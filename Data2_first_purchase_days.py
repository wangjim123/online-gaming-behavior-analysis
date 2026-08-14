import matplotlib.pyplot as plt
import pandas as pd
import os

def main(df1):
    df = df1.copy()
    #
    # #-----------------------------------------------------------
    # # 真實付費人數  未課金/無金額紀錄玩家數
    # # 2. 定義真實付費玩家標籤 (金額不為空且 > 0)
    # df["IsPayer"] = df["InAppPurchaseAmount"].notnull() & (
    #         df["InAppPurchaseAmount"] > 0
    # )
    #
    # # 基礎數據統計
    # total_users = len(df)
    # paying_users = df["IsPayer"].sum()
    # non_paying_users = total_users - paying_users
    # overall_conversion_rate = paying_users / total_users
    #
    # print("=== 📊 修正前的資料分析 ===")
    # print(f"總玩家數: {total_users}")
    # print(
    #     f"真實付費玩家數 (InAppPurchaseAmount > 0): {paying_users} (付費率:"
    #     f" {overall_conversion_rate:.2%})"
    # )
    # print(
    #     f"未課金 / 無金額紀錄玩家數: {non_paying_users}"
    #     f" ({non_paying_users / total_users:.2%})\n"
    # )
    #
    # #-----------------------------------------------------------

    # ----------------------------------------------------------
    # 定義付費標記 (1: 有課金, 0: 未課金)
    df["IsPayer"] = df["InAppPurchaseAmount"].notnull() & (
            df["InAppPurchaseAmount"] > 0
    )
    paying_users = df["IsPayer"].sum()

    # 篩選 0~30 天首課玩家
    days_df = df[
        df["IsPayer"]
        & (df["FirstPurchaseDaysAfterInstall"] >= 0)
        & (df["FirstPurchaseDaysAfterInstall"] <= 30)
        ].copy()

    # 計算每日首課人數 (補齊 0~30 天完整索引)
    daily_counts = days_df[
        "FirstPurchaseDaysAfterInstall"
    ].value_counts().sort_index()
    full_days = pd.Series(0, index=range(0, 31)) # 確保沒有缺值NAN
    daily_counts = full_days.add(daily_counts, fill_value=0)

    # 計算每日佔比與累計佔比 (%)
    daily_pct = (daily_counts / paying_users) * 100 # 每日占比
    daily_cumulative = daily_counts.cumsum() # 累計人數
    daily_cum_pct_payers = (daily_cumulative / paying_users) * 100

    # ==========================================
    # 2. 建立圖表畫布 (1 列 2 欄)
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
    # --------------------------------------------
    # 左圖：每日首課人數佔比 (Daily Share %)
    ax1.plot(
        daily_pct.index,
        daily_pct.values,
        marker="o",
        color="#1f77b4",
        linewidth=2.2,
        markersize=5.5,
        label="Daily Share (%)",
    )
    ax1.fill_between(
        daily_pct.index, daily_pct.values, 2.0, color="#1f77b4", alpha=0.12
    )

    # 繪製平均基準線
    avg_pct = daily_pct.mean()
    ax1.axhline(
        avg_pct,
        color="#e74c3c",
        linestyle="--",
        alpha=0.8,
        label=f"Daily Average ({avg_pct:.2f}%)",
    )

    # 標註最高點與最低點
    max_day, max_val = daily_pct.idxmax(), daily_pct.max()
    min_day, min_val = daily_pct.idxmin(), daily_pct.min()
    ax1.scatter(
        [max_day, min_day],
        [max_val, min_val],
        color=["#2ca02c", "#d62728"],
        s=70,
        zorder=5,
    )
    # 標點內容 位置
    ax1.annotate(
        f"Peak: Day {max_day} ({max_val:.2f}%)",
        xy=(max_day, max_val),
        xytext=(max_day - 1, max_val + 0.12),
        fontweight="bold",
        fontsize=9.5,
        color="#2ca02c",
    )
    ax1.annotate(
        f"Low: Day {min_day} ({min_val:.2f}%)",
        xy=(min_day, min_val),
        xytext=(min_day - 1, min_val - 0.18),
        fontweight="bold",
        fontsize=9.5,
        color="#d62728",
    )
    # 標題
    ax1.set_title(
        "Daily First Purchase Share (% of Payers)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax1.set_xlabel("Days After Install (Day)", fontsize=11)
    ax1.set_ylabel("Daily Share (%)", fontsize=11)
    ax1.set_xticks(range(0, 31, 2))
    ax1.set_ylim(2.0, 4.2)  # 刪除 0~2.0 無資料區間，凸顯波動
    ax1.legend(loc="upper right", frameon=True)
    # ==========================================
    # -------------------------------------------------------------
    # 右圖：累計付費玩家首課佔比 (Cumulative Conversion Share %)
    # -------------------------------------------------------------
    ax2.plot(
        daily_cum_pct_payers.index,
        daily_cum_pct_payers.values,
        marker="s",
        color="#e67e22",
        linewidth=2.5,
        markersize=5,
    )
    ax2.set_title(
        "Cumulative Conversion Share of Payers (%)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax2.set_xlabel("Days After Install (Day)", fontsize=11)
    ax2.set_ylabel("Cumulative Share (%)", fontsize=11)
    ax2.set_xticks(range(0, 31, 2))
    ax2.set_yticks(range(0, 101, 10))
    ax2.set_ylim(0, 105)

    # 標註關鍵里程碑 (Day 1, 7, 14, 30)
    milestones = [
        (1, "#e74c3c"),
        (7, "#f39c12"),
        (14, "#27ae60"),
        (30, "#8e44ad"),
    ]
    for day, color in milestones:
        val = daily_cum_pct_payers[day]
        ax2.scatter(day, val, color=color, s=75, zorder=5)
        ax2.annotate(
            f"Day {day}: {val:.1f}%",
            xy=(day, val),
            xytext=(
                day - 3 if day > 20 else day + 1,
                val - 6 if day > 20 else val + 2.5,
            ),
            fontweight="bold",
            fontsize=9.5,
            color=color,
        )
    # 標題
    plt.tight_layout()
    # 存檔
    os.makedirs("Data2_png", exist_ok=True)
    output_path = os.path.join("Data2_png", "Data2_first_purchase_days.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #---------------------------------------------------------
    plt.show()

    # 設定首課時間區間 (Timing Bins)
    # Day 0-1 (首日首課), Day 2-7 (首週首課), Day 8-14, Day 15-30, >30天或未課金
    bins = [-1, 1, 7, 14, 30, 999]
    labels = ["Day 0-1", "Day 2-7", "Day 8-14", "Day 15-30", "Unconverted"]


    # 計算關鍵指標 (轉化率與提升倍數 Lift Factor)
    total_users = len(df)
    paying_users = df["IsPayer"].sum()
    overall_conversion_rate = paying_users / total_users

    # 累計前 N 天完成首課的付費玩家數
    d1_payers = (df["FirstPurchaseDaysAfterInstall"] <= 1).sum()
    d7_payers = (df["FirstPurchaseDaysAfterInstall"] <= 7).sum() - d1_payers
    d14_payers = (df["FirstPurchaseDaysAfterInstall"] <= 14).sum() - d1_payers - d7_payers

    d1_conversion_share = d1_payers / paying_users
    d7_conversion_share = d7_payers / paying_users
    d14_conversion_share = d14_payers / paying_users

    print("=== 📊 首課天數與轉化率分析 ===")
    print(f"總玩家數: {total_users}")
    print(f"總付費玩家數: {paying_users} (整體付費率: {overall_conversion_rate:.2%})\n")

    print(
        f"• 第 1 天內完成首課玩家數: {d1_payers} 人 (占總付費玩家 {d1_conversion_share:.2%})"
    )
    print(
        f"• 第 2~7 天內完成首課玩家數: {d7_payers} 人 (占總付費玩家 {d7_conversion_share:.2%})\n"
    )
    print(
        f"• 第 8~14 天內完成首課玩家數: {d14_payers} 人 (占總付費玩家 {d14_conversion_share:.2%})\n"
    )
