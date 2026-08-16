import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report, roc_auc_score, average_precision_score, f1_score
import lightgbm as lgb
import os

def main(df1):
    df = df1.copy()

    # 定義流失目標（Target Variable）：以 EngagementLevel == 'Low' 為流失標籤
    # 代表高流失風險 (Churn), 0 代表穩定留存 (Retain)
    df['Churn'] = (df['EngagementLevel'] == 'Low').astype(int)

    # 移除對預測無效益的唯一識別碼與原始標籤
    df["TotalPlayTime"] = df["AvgSessionDurationMinutes"] * df["SessionsPerWeek"]
    X = df.drop(columns=["SessionsPerWeek", "AvgSessionDurationMinutes",
                         'PlayerID', 'EngagementLevel', 'Churn', "PlayTimeHours"
                        ], errors='ignore')
    y = df['Churn']

    # print("\n目標變數分布：")
    # print(y.value_counts(normalize=True))

    # 特徵型態轉換 (LightGBM 原生類別處理)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 建立與訓練 LightGBM 模型
    # 使用 is_unbalance=True 自動應對少數派流失玩家的不平衡問題
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        is_unbalance=True,
        verbose=-1
    )
    # 訓練模型
    model.fit(X_train, y_train)
    train(X_train, X_test, y_train, y_test, model)
    churn_analysis(X_test, df1, model)

def train(X_train, X_test, y_train, y_test, model):

    # 4. 預測類別與概率 & 模型評估
    # ==========================================
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== LightGBM 模型評估結果 ===")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}\n")
    print(f"F1-Score : {f1_score(y_test, y_pred, pos_label=1):.4f} (預設門檻 0.5 下的流失抓取綜合分數)\n")

    print(classification_report(y_test, y_pred, target_names=['Retain (0)', 'Churn (1)']))

    # 混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩陣 (Confusion Matrix):")
    print(f"真正留存 (TN): {cm[0, 0]} | 誤判流失 (FP): {cm[0, 1]}")
    print(f"漏抓流失 (FN): {cm[1, 0]} | 成功預警 (TP): {cm[1, 1]}")

    # 5. 繪製 LightGBM 特徵重要性圖表
    # ==========================================
    # 直接從 LightGBM 提取特徵重要性 (不需要重新拼接 One-Hot 欄位名稱)
    feature_imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    feature_imp.tail(10).plot(kind='barh', color='#2b5c8f')
    plt.title('Top 10 Feature Importances (LightGBM Churn Model)')
    plt.xlabel('Split Importance (Frequency of Feature Split)')
    # 標題
    plt.tight_layout()
    # 存檔
    os.makedirs("Data1_png", exist_ok=True)
    output_path = os.path.join("Data1_png", "Data1_Top_10_Feature_Importances_(LightGBM_Churn_Model).png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    #---------------------------------------------------------
    plt.show()

def churn_analysis(X_test, df, model):
    # 🎯 核心：計算測試集玩家的流失機率 (%)
    churn_probabilities = model.predict_proba(X_test)[:, 1] * 100

    # 修正：使用 X_test.index 只對齊測試集的原始資料！
    result_df = df.loc[X_test.index].copy()
    result_df['Churn_Probability_%'] = np.round(churn_probabilities, 2)

    # 根據 % 數劃分風險等級
    conditions = [
        (result_df['Churn_Probability_%'] >= 70),
        (result_df['Churn_Probability_%'] >= 30) & (result_df['Churn_Probability_%'] < 70),
        (result_df['Churn_Probability_%'] < 30)
    ]
    choices = ['高風險 (>70%)', '中風險 (30-70%)', '低風險 (<30%)']

    result_df['Risk_Level'] = np.select(conditions, choices, default='未知')

    # 📊 檢視結果與篩選名單
    print("=== 測試集各風險等級玩家分布 ===")
    print(result_df['Risk_Level'].value_counts())

    print("\n=== 前 5 筆測試集玩家流失機率範例 ===")
    print(result_df[['PlayerID', 'EngagementLevel', 'Churn_Probability_%', 'Risk_Level']].head())

    # 🚀 篩選高風險玩家名單並匯出
    high_risk_players = result_df[result_df['Churn_Probability_%'] > 70]

    print(f"\n[警報] 測試集中成功篩選出 {len(high_risk_players)} 位高風險玩家（流失機率 > 70%）")
    #
    # # 檢查欄位是否存在後才匯出，避免 KeyError
    # export_cols = [col for col in ['PlayerID', 'Churn_Probability_%', 'PlayTimeHours', 'InGamePurchases'] if col in high_risk_players.columns]
    # high_risk_players[export_cols].to_csv('high_risk_churn_players_top70pct.csv', index=False)
    # print("名單已成功匯出至 'high_risk_churn_players_top70pct.csv'！\n")