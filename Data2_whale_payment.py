import matplotlib.pyplot as plt
import os

def main(df1):
    df = df1.copy()

    # 1. 篩選鯨魚玩家資料
    payer_df = df[
        df["InAppPurchaseAmount"].notnull() & (df["InAppPurchaseAmount"] > 0)
        ].copy()
    whales_df = payer_df[payer_df["SpendingSegment"] == "Whale"]

    # 2. 計算鯨魚玩家支付管道分佈
    whale_payment = whales_df["PaymentMethod"].value_counts()

    # 3. 繪製圓餅圖 (圖 4)
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.figure(figsize=(7, 6), dpi=120)

    colors_pie = [
        "#3498db",
        "#2ecc71",
        "#e74c3c",
        "#f1c40f",
        "#9b59b6",
        "#34495e",
        "#1abc9c",
    ]
    wedges, texts, autotexts = plt.pie(
        whale_payment.values,
        labels=whale_payment.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors_pie[: len(whale_payment)],
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    plt.setp(autotexts, size=9.5, weight="bold", color="white")
    plt.setp(texts, size=10, weight="bold")
    plt.title(
        "4. Whale Payment Method Distribution",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    plt.tight_layout()
    # 存檔
    os.makedirs("Data2_png", exist_ok=True)
    output_path = os.path.join("Data2_png", "Data2_whale_payment.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #---------------------------------------------------------
    plt.show()