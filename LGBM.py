import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plot_feature_importance

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
            n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        print(f"=== [{exp_name}] ===")
        print(f"使用特徵數: {X_tr.shape[1]}")
        print(f"Accuracy: {acc:.4f}\n")

        return model, acc

    # 實驗 1：全部原始特徵 (Baseline)
    X_train_A = X_train.copy()
    X_test_A = X_test.copy()
    model_A, acc_A = evaluate_lgbm(
        X_train_A, X_test_A, y_train, y_test, exp_name="組合 A：全部原始特徵"
    )
    plot_feature_importance.main(
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
    model_B, acc_B = evaluate_lgbm(
        X_train_B, X_test_B, y_train, y_test, exp_name="組合 B：新增黏著度特徵"
    )

    plot_feature_importance.main(
        model_B, X_train_B.columns, "Feature Importance - 組合 B (新增黏著度特徵)"
    )

    # 實驗 3：刪除重要性較低特徵 (Feature Selection)
    importance = model_B.feature_importances_
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

    model_C, acc_C = evaluate_lgbm(
        X_train_C, X_test_C, y_train, y_test, exp_name="組合 C：刪除低重要性特徵"
    )
    plot_feature_importance.main(
        model_C, X_train_C.columns, "Feature Importance - 組合 C (刪除低重要性特徵)"
    )

    # 3. 彙總三者結果比較
    summary_df = pd.DataFrame(
        {
            "實驗組合": ["組合 A (原始)", "組合 B (新增黏著度)", "組合 C (刪除低重要性)"],
            "特徵數量": [
                X_train_A.shape[1],
                X_train_B.shape[1],
                X_train_C.shape[1],
            ],
            "Accuracy": [acc_A, acc_B, acc_C],
        }
    )

    print("=== 最終實驗結果彙總 ===")
    print(summary_df.to_string(index=False))