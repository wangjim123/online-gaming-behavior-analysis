import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import os

def main(df1):
    df = df1.copy()

    # 1. 讀取資料與定義流失目標 (以最後課金時間計算)
    df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'], errors='coerce')
    snapshot_date = df['LastPurchaseDate'].max()
    days_since = (snapshot_date - df['LastPurchaseDate']).dt.days

    # 超過 60 天未課金視為流失 (1)，60 天以內為留存 (0)
    df['Churn'] = ((days_since.fillna(999)) > 60).astype(int)

    # 2. 特徵前處理 (僅使用原始特徵，避免資料洩漏)
    num_cols = [
        'Age',
        'SessionCount',
        'AverageSessionLength',
        'InAppPurchaseAmount',
        'FirstPurchaseDaysAfterInstall',
    ]
    cat_cols = [
        'Gender',
        'Country',
        'Device',
        'GameGenre',
        'SpendingSegment',
        'PaymentMethod',
    ]

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')

    cat_encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    )
    df_encoded = df.copy()
    df_encoded[cat_cols] = cat_encoder.fit_transform(df[cat_cols])

    X = df_encoded[num_cols + cat_cols]
    y = df['Churn']

    # 3. 訓練梯度提升樹 (GBM)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cat_mask = [c in cat_cols for c in X.columns]

    model_gbm = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        categorical_features=cat_mask,
        class_weight='balanced',
        random_state=42,
    )
    model_gbm.fit(X_train, y_train)

    # 4. 預測流失機率與構建策略矩陣
    df['Churn_Prob'] = model_gbm.predict_proba(X)[:, 1]
    val_threshold = df['InAppPurchaseAmount'].quantile(0.75)
    df['Is_High_Value'] = (df['SpendingSegment'].isin(['Whale', 'Dolphin'])) | (
            df['InAppPurchaseAmount'] >= val_threshold
    )
    df['Is_High_Risk'] = df['Churn_Prob'] >= 0.70

    def assign_strategy(row):
        if row['Is_High_Value'] and row['Is_High_Risk']:
            return (
                'High Value x High Risk\n(Core Retention: Exclusive Gift Package)'
            )
        elif row['Is_High_Value'] and not row['Is_High_Risk']:
            return 'High Value x Low Risk\n(VIP Loyalty: Premium Rewards)'
        elif not row['Is_High_Value'] and row['Is_High_Risk']:
            return 'Low Value x High Risk\n(Light Re-engagement: Promo Pack)'
        else:
            return 'Low Value x Low Risk\n(Daily Operation: General Push)'

    df['Action_Strategy'] = df.apply(assign_strategy, axis=1)

    # 5. 繪製並儲存四合一圖表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    palette = {
        'High Value x High Risk\n(Core Retention: Exclusive Gift Package)': (
            '#E74C3C'
        ),
        'High Value x Low Risk\n(VIP Loyalty: Premium Rewards)': '#2ECC71',
        'Low Value x High Risk\n(Light Re-engagement: Promo Pack)': '#F39C12',
        'Low Value x Low Risk\n(Daily Operation: General Push)': '#3498DB',
    }

    # (1) 散佈圖
    sns.scatterplot(
        data=df,
        x='Churn_Prob',
        y='InAppPurchaseAmount',
        hue='Action_Strategy',
        palette=palette,
        alpha=0.6,
        s=40,
        ax=axes[0, 0],
    )
    axes[0, 0].axvline(
        0.70, color='red', linestyle='--', alpha=0.7, label='Risk Threshold (0.70)'
    )
    axes[0, 0].axhline(
        val_threshold,
        color='green',
        linestyle='--',
        alpha=0.7,
        label=f'Value Threshold (${val_threshold:.1f})',
    )
    axes[0, 0].set_title(
        '1. Value vs. Churn Risk Matrix (Gift Package Targets)',
        fontsize=13,
        fontweight='bold',
    )
    axes[0, 0].set_xlabel('Predicted Churn Probability')
    axes[0, 0].set_ylabel('In-App Purchase Amount ($)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend(fontsize=8, loc='upper left')

    # (2) 分群人數長條圖
    counts = df['Action_Strategy'].value_counts()
    bars = axes[0, 1].barh(
        counts.index, counts.values, color=[palette[k] for k in counts.index]
    )
    axes[0, 1].set_title(
        '2. Player Segment Population Distribution', fontsize=13, fontweight='bold'
    )
    axes[0, 1].set_xlabel('Player Count')
    for bar in bars:
        w = bar.get_width()
        axes[0, 1].text(
            w + 15,
            bar.get_y() + bar.get_height() / 2,
            f'{int(w)} ({w / len(df) * 100:.1f}%)',
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold',
        )
    axes[0, 1].set_xlim(0, max(counts.values) * 1.2)

    # (3) ROC 曲線
    y_test_prob = model_gbm.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    axes[1, 0].plot(
        fpr,
        tpr,
        color='#8E44AD',
        lw=2.5,
        label=f'GBM ROC Curve (AUC = {auc(fpr, tpr):.3f})',
    )
    axes[1, 0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
    axes[1, 0].set_title(
        '3. Model ROC-AUC Curve (Test Set)', fontsize=13, fontweight='bold'
    )
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].grid(True, alpha=0.3)

    # (4) 混淆矩陣
    cm = confusion_matrix(y_test, model_gbm.predict(X_test))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        ax=axes[1, 1],
        xticklabels=['Predicted Retain (0)', 'Predicted Churn (1)'],
        yticklabels=['Actual Retain (0)', 'Actual Churn (1)'],
    )
    axes[1, 1].set_title(
        '4. Confusion Matrix (Test Set)', fontsize=13, fontweight='bold'
    )

    # 標題
    plt.tight_layout()
    # 存檔
    os.makedirs("Data2_png", exist_ok=True)
    output_path = os.path.join("Data2_png", "Data2_churn_value_strategy_analysis.png.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #---------------------------------------------------------
    plt.show()

    # 6. 匯出 393 位核心禮包投放玩家名單
    target_export = df[df['Action_Strategy'].str.contains('Core Retention')][[
        'UserID',
        'SpendingSegment',
        'InAppPurchaseAmount',
        'SessionCount',
        'AverageSessionLength',
        'FirstPurchaseDaysAfterInstall',
        'PaymentMethod',
        'Churn_Prob',
    ]]
    target_export.to_csv('core_gift_package_targets.csv', index=False)
    print(f'已成功匯出 {len(target_export)} 位目標玩家名單！')