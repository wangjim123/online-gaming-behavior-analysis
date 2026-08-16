import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
def main(model, feature_names, title):
    # --- 解決 Matplotlib 中文顯示與負號顯示問題 ---
    plt.rcParams['font.sans-serif'] = [
        'Microsoft JhengHei',  # Windows 微軟正黑體
        'Arial Unicode MS',  # Mac 預設中文字型
        'SimHei',  # Linux / 常見黑體
        'sans-serif',
    ]
    plt.rcParams['axes.unicode_minus'] = (
        False  # 修正負號 (-) 變成方塊或無法顯示的問題
    )
    """繪製單一模型的特徵重要性圖表"""
    importance = model.booster_.feature_importance(importance_type="gain")
    df_imp = (
        pd.DataFrame({"Feature": feature_names, "Importance": importance})
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    # 3. 繪圖
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=df_imp,
        palette="viridis",
        hue="Feature",
        legend=False,
    )

    # 4. 更新標題與軸標籤 (標示為 Gain)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Importance (Total Gain)", fontsize=12)  # <-- 改成 Gain
    plt.ylabel("Features", fontsize=12)
    # 標題
    plt.tight_layout()
    # 存檔
    os.makedirs("Data1_png", exist_ok=True)
    output_path = os.path.join("Data1_png", f"Data1{title}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #---------------------------------------------------------
    plt.show()