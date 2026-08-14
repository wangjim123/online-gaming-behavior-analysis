import matplotlib.pyplot as plt
import pandas as pd


def main(df1):
    df = df1.copy()
    # 1. 篩選鯨魚玩家資料
    payer_df = df[
        df["InAppPurchaseAmount"].notnull() & (df["InAppPurchaseAmount"] > 0)
        ].copy()
    whales_df = payer_df[payer_df["SpendingSegment"] == "Whale"]

    # 2. 計算鯨魚偏好的遊戲品類排行 (TOP 6)
    whale_genres = (
        whales_df.groupby("GameGenre")["InAppPurchaseAmount"]
        .agg(["count", "sum"])
        .sort_values(by="sum", ascending=True)
        .tail(6)
    )

    # 3. 繪製橫向長條圖 (圖 3)
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.figure(figsize=(9, 5.5), dpi=300)

    plt.barh(
        whale_genres.index,
        whale_genres["sum"] / 1000,
        color="#e67e22",
        edgecolor="black",
        height=0.6,
        alpha=0.85,
    )
    plt.title(
        "3. Top Game Genres for Whales (Spend in $k USD)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Total Whale Spend ($1,000 USD)", fontsize=11)
    plt.xlim(0, (whale_genres["sum"].max() / 1000) * 1.25)

    for i, (genre, val) in enumerate(
            zip(whale_genres.index, whale_genres["sum"] / 1000)
    ):
        cnt = whale_genres.loc[genre, "count"]
        plt.annotate(
            f"${val:.1f}k ({cnt} Whales)",
            xy=(val, i),
            xytext=(5, -3),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()