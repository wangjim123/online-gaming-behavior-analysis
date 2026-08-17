import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import Data1_plot_feature_importance

def main(df1):
    df = df1.copy()

    #特徵和目標處理
    X = df.drop(columns=["PlayerID", "EngagementLevel"])
    y = df["EngagementLevel"]

    # 找出所有 object 型態的欄位並轉為 category
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')

    # 切分訓練集與測試集 (80% 訓練 / 20% 測試)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 建立單一評估模型函式
    def evaluate_lgbm(X_tr, X_te, y_tr, y_te, exp_name="實驗"):
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            importance_type='gain',
            random_state=42,
            verbose=-1
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        # 取得各特徵 Gain 重要性
        gains = model.booster_.feature_importance(importance_type='gain')
        df_imp = pd.DataFrame(
            {'feature': X_tr.columns, 'gain': gains}
        ).sort_values(by='gain', ascending=False)
        df_imp['gain_pct'] = (df_imp['gain'] / df_imp['gain'].sum()) * 100

        return model, acc, df_imp

    # 實驗 1：全部原始特徵 (Baseline)
    X_train_A = X_train.copy()
    X_test_A = X_test.copy()
    model_A, acc_A, df_imp_A = evaluate_lgbm(
        X_train_A, X_test_A, y_train, y_test, exp_name="組合 A：全部原始特徵"
    )
    Data1_plot_feature_importance.main(
        model_A, X_train_A.columns, "Feature Importance -組合 A (全部原始特徵)"
    )


    # 實驗 2：新增黏著度特徵 (Feature Engineering)
    X_train_B = X_train.copy()
    X_test_B = X_test.copy()
    X_train_B["WeeklyTotalMinutes"] = (
            X_train_B["AvgSessionDurationMinutes"] * X_train_B["SessionsPerWeek"]
    )
    X_test_B["WeeklyTotalMinutes"] = (
            X_test_B["AvgSessionDurationMinutes"] * X_test_B["SessionsPerWeek"]
    )
    model_B, acc_B, df_imp_B = evaluate_lgbm(
        X_train_B, X_test_B, y_train, y_test, exp_name="組合 B：新增黏著度特徵"
    )

    Data1_plot_feature_importance.main(
        model_B, X_train_B.columns, "Feature Importance - 組合 B (新增黏著度特徵)"
    )

    # 實驗 3：刪除重要性較低特徵 (Feature Selection)
    importance = model_B.booster_.feature_importance(importance_type="gain")
    feature_names = X_train_B.columns

    df_imp = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values(by="importance", ascending=False)

    # 2. 設定篩選門檻 (例如：刪除重要性分布最後 20% 的特徵，或重要性 <= 5 的特徵)
    # 這裡範例為：保留重要性前 80% 的特徵 (剔除最不重要的 20%)
    keep_count = int(len(df_imp) * 0.5)
    top_features = df_imp.iloc[:keep_count]["feature"].tolist()

    X_train_C = X_train_B[top_features]
    X_test_C = X_test_B[top_features]

    model_C, acc_C, df_imp_C = evaluate_lgbm(
        X_train_C, X_test_C, y_train, y_test, exp_name="組合 C：刪除低重要性特徵"
    )
    Data1_plot_feature_importance.main(
        model_C, X_train_C.columns, "Feature Importance - 組合 C (刪除低重要性特徵)"
    )

    # 3. 計算進階指標
    top1_A = df_imp_A.iloc[0]
    top1_B = df_imp_B.iloc[0]
    top1_C = df_imp_C.iloc[0]

    # 核心資訊保留率 (前 6 個特徵在組合 B 總 Gain 中的佔比)
    info_retention_pct = (
                                 df_imp_B.iloc[:keep_count]['gain'].sum() / df_imp_B['gain'].sum()
                         ) * 100

    # 輸出 KPI 卡片
    print('========================================')
    print('       三組特徵工程綜合對比分析       ')
    print('========================================')
    print(f'【特徵數量精簡】: {len(df_imp_A)} → {len(df_imp_C)} 項')
    print(f'【最高特徵重要性佔比】: {top1_B["gain_pct"]:.1f}%')
    print(f'【核心資訊保留率】: {info_retention_pct:.1f}%\n')

    # 輸出綜合對比表格
    summary_table = pd.DataFrame([
        {
            '組合名稱': '組合 A (原始特徵)',
            '特徵數量': f'{len(df_imp_A)} 項',
            'Accuracy': f'{acc_A * 100:.2f}%',
            'Top 1 主導特徵': top1_A['feature'],
            'Top 1 Total Gain': f'{top1_A["gain"]:,.0f}',
            'Top 1 Gain 佔比': f'{top1_A["gain_pct"]:.1f}%',
            '特徵工程策略評估': '基準模型：訊號分散於單次時長與每週次數',
        },
        {
            '組合名稱': '組合 B (新增黏著特徵)',
            '特徵數量': f'{len(df_imp_B)} 項',
            'Accuracy': f'{acc_B * 100:.2f}%',
            'Top 1 主導特徵': top1_B['feature'],
            'Top 1 Total Gain': f'{top1_B["gain"]:,.0f}',
            'Top 1 Gain 佔比': f'{top1_B["gain_pct"]:.1f}%',
            '特徵工程策略評估': '訊號聚合：成功捕捉黏著度複合特徵',
        },
        {
            '組合名稱': '組合 C (刪除無效益特徵)',
            '特徵數量': f'{len(df_imp_C)} 項',
            'Accuracy': f'{acc_C * 100:.2f}%',
            'Top 1 主導特徵': top1_C['feature'],
            'Top 1 Total Gain': f'{top1_C["gain"]:,.0f}',
            'Top 1 Gain 佔比': f'{top1_C["gain_pct"]:.1f}%',
            '特徵工程策略評估': '最佳推薦：輕量、高效、高可解釋性最佳配置',
        },
    ])

    print(summary_table.to_string(index=False))