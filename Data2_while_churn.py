import os
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def main(df1):
    df = df1.copy()

    # =============================================================
    # 1. 資料前處理
    # =============================================================
    df_clean = df.dropna(
        subset=[
            'Age',
            'Country',
            'GameGenre',
            'SpendingSegment',
            'SessionCount',
            'AverageSessionLength',
            'FirstPurchaseDaysAfterInstall',
            'PaymentMethod',
        ]
    ).copy()

    # 定義鯨魚標籤
    df_clean['IsWhale'] = (df_clean['SpendingSegment'] == 'Whale').astype(int)

    # 特徵編碼
    features = [
        "Age",
        "SessionCount",
        "AverageSessionLength",
        "FirstPurchaseDaysAfterInstall",
        "Country",
        "GameGenre",
        "PaymentMethod",
    ]
    cat_features = ["Country", "GameGenre", "PaymentMethod"]

    for col in cat_features:
        df_clean[col] = df_clean[col].astype("category")

    X = df_clean[features]
    y = df_clean["IsWhale"]

    # 切出獨立測試集 (Holdout Test Set 20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 正負樣本比例加權
    scale_pos_weight = (len(y_train_val) - y_train_val.sum()) / y_train_val.sum()

    # =============================================================
    # 2. [新增] 不使用 CV 的直接訓練模型 (Baseline: 門檻 0.50)
    # =============================================================
    print("=========================================================")
    print("🚀 [方法一] 不使用 CV：直接訓練並以預設門檻 (0.50) 評估")
    print("=========================================================")
    direct_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )
    direct_model.fit(X_train_val, y_train_val)

    # 預測
    dir_train_proba = direct_model.predict_proba(X_train_val)[:, 1]
    dir_test_proba = direct_model.predict_proba(X_test)[:, 1]
    dir_train_pred = (dir_train_proba >= 0.50).astype(int)
    dir_test_pred = (dir_test_proba >= 0.50).astype(int)

    # 評估指標
    dir_train_f1 = f1_score(y_train_val, dir_train_pred, zero_division=0)
    dir_test_f1 = f1_score(y_test, dir_test_pred, zero_division=0)
    dir_train_rec = recall_score(y_train_val, dir_train_pred, zero_division=0)
    dir_test_rec = recall_score(y_test, dir_test_pred, zero_division=0)
    dir_train_prec = precision_score(y_train_val, dir_train_pred, zero_division=0)
    dir_test_prec = precision_score(y_test, dir_test_pred, zero_division=0)

    print(f"訓練集 (Train) -> F1: {dir_train_f1:.4f} | Precision: {dir_train_prec:.4f} | Recall: {dir_train_rec:.4f}")
    print(f"測試集 (Test)  -> F1: {dir_test_f1:.4f} | Precision: {dir_test_prec:.4f} | Recall: {dir_test_rec:.4f}\n")

    # =============================================================
    # 3. 使用 5-Fold CV 尋找最佳決策門檻 (Threshold Tuning)
    # =============================================================
    print("=========================================================")
    print("🔄 [方法二] 使用 5-Fold CV 搜尋最佳決策門檻")
    print("=========================================================")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train_val))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
        X_tr, y_tr = X_train_val.iloc[train_idx], y_train_val.iloc[train_idx]
        X_va, y_va = X_train_val.iloc[val_idx], y_train_val.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            scale_pos_weight=scale_pos_weight,
            random_state=42 + fold,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]

    # 搜尋最佳 F1-score 決策門檻
    thresholds = np.linspace(0.01, 0.99, 100)
    cv_f1_scores, cv_precisions, cv_recalls = [], [], []

    for t in thresholds:
        y_pred_t = (oof_preds >= t).astype(int)
        cv_f1_scores.append(f1_score(y_train_val, y_pred_t, zero_division=0))
        cv_precisions.append(precision_score(y_train_val, y_pred_t, zero_division=0))
        cv_recalls.append(recall_score(y_train_val, y_pred_t, zero_division=0))

    best_idx = np.argmax(cv_f1_scores)
    best_threshold = thresholds[best_idx]
    best_cv_f1 = cv_f1_scores[best_idx]
    best_cv_prec = cv_precisions[best_idx]
    best_cv_rec = cv_recalls[best_idx]

    print(f'最佳 CV 決策閥值 (Best Threshold) : {best_threshold:.3f}')
    print(f'驗證集最佳 F1-Score              : {best_cv_f1:.4f}')
    print(f'驗證集對應 Precision             : {best_cv_prec:.4f}')
    print(f'驗證集對應 Recall                : {best_cv_rec:.4f}\n')

    # =============================================================
    # 4. 全量訓練 (CV 最佳化後) 並在訓練集與測試集上評估
    # =============================================================
    final_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )
    final_model.fit(X_train_val, y_train_val)

    train_pred_proba = final_model.predict_proba(X_train_val)[:, 1]
    test_pred_proba = final_model.predict_proba(X_test)[:, 1]

    # --- (A) 訓練集評估 (Train Scores) ---
    print("=========================================================")
    print("📊 訓練集 (Train Set) 表現評估")
    print("=========================================================")
    # 預設 0.50 門檻
    y_train_def = (train_pred_proba >= 0.50).astype(int)
    train_f1_def = f1_score(y_train_val, y_train_def, zero_division=0)
    train_prec_def = precision_score(y_train_val, y_train_def, zero_division=0)
    train_rec_def = recall_score(y_train_val, y_train_def, zero_division=0)

    # CV 最佳門檻
    y_train_opt = (train_pred_proba >= best_threshold).astype(int)
    train_f1_opt = f1_score(y_train_val, y_train_opt, zero_division=0)
    train_prec_opt = precision_score(y_train_val, y_train_opt, zero_division=0)
    train_rec_opt = recall_score(y_train_val, y_train_opt, zero_division=0)

    print(f"1. 訓練集預設門檻 (0.50)           -> F1: {train_f1_def:.4f} | Precision: {train_prec_def:.4f} | Recall: {train_rec_def:.4f}")
    print(f"2. 訓練集套用最佳門檻 ({best_threshold:.2f}) -> F1: {train_f1_opt:.4f} | Precision: {train_prec_opt:.4f} | Recall: {train_rec_opt:.4f}\n")

    # --- (B) 測試集評估 (Test Scores) ---
    print("=========================================================")
    print("🎯 測試集 (Test Set) 表現評估")
    print("=========================================================")
    # 預設 0.50 門檻
    y_test_def = (test_pred_proba >= 0.50).astype(int)
    test_f1_def = f1_score(y_test, y_test_def, zero_division=0)
    test_prec_def = precision_score(y_test, y_test_def, zero_division=0)
    test_rec_def = recall_score(y_test, y_test_def, zero_division=0)

    # CV 最佳門檻
    y_test_opt = (test_pred_proba >= best_threshold).astype(int)
    test_f1_opt = f1_score(y_test, y_test_opt, zero_division=0)
    test_prec_opt = precision_score(y_test, y_test_opt, zero_division=0)
    test_rec_opt = recall_score(y_test, y_test_opt, zero_division=0)

    print(f"1. 測試集預設門檻 (0.50)           -> F1: {test_f1_def:.4f} | Precision: {test_prec_def:.4f} | Recall: {test_rec_def:.4f}")
    print(f"2. 測試集套用最佳門檻 ({best_threshold:.2f}) -> F1: {test_f1_opt:.4f} | Precision: {test_prec_opt:.4f} | Recall: {test_rec_opt:.4f}\n")

    # =============================================================
    # 5. 視覺化：CV 搜尋曲線 + 訓練/測試集成效對照
    # =============================================================
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=120)

    # 左圖: CV 驗證集門檻搜尋曲線
    ax1.plot(thresholds, cv_f1_scores, label="CV F1-Score", color="#e74c3c", linewidth=2.5)
    ax1.plot(thresholds, cv_recalls, label="CV Recall", color="#2ecc71", linewidth=2, linestyle="--")
    ax1.plot(thresholds, cv_precisions, label="CV Precision", color="#3498db", linewidth=2, linestyle=":")
    ax1.axvline(best_threshold, color="red", linestyle="--", alpha=0.7, label=f"Best Threshold ({best_threshold:.2f})")
    ax1.scatter(best_threshold, best_cv_f1, color="red", s=80, zorder=5)

    ax1.set_title("1. Cross-Validation: Threshold Tuning Curve", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Decision Threshold", fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)

    # 右圖: 測試集指標對比 (固定 0.50 vs CV 最佳門檻)
    labels = ["Precision", "Recall", "F1-Score"]
    default_metrics = [test_prec_def, test_rec_def, test_f1_def]
    opt_metrics = [test_prec_opt, test_rec_opt, test_f1_opt]

    x = np.arange(len(labels))
    width = 0.35

    ax2.bar(x - width / 2, default_metrics, width, label="Fixed Threshold (0.50)", color="#95a5a6", edgecolor="black", alpha=0.85)
    ax2.bar(x + width / 2, opt_metrics, width, label=f"CV Optimal Threshold ({best_threshold:.2f})", color="#e74c3c", edgecolor="black", alpha=0.85)

    ax2.set_title("2. Test Set Evaluation: Fixed vs. CV Optimal Threshold", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Metric Value", fontsize=11)
    ax2.set_ylim(0, 1.15)
    ax2.legend(loc="upper right", frameon=True)

    for idx, (v_def, v_opt) in enumerate(zip(default_metrics, opt_metrics)):
        ax2.annotate(f"{v_def:.2f}", xy=(idx - width / 2, v_def), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax2.annotate(f"{v_opt:.2f}", xy=(idx + width / 2, v_opt), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#c0392b")

    plt.tight_layout()
    os.makedirs("Data2_png", exist_ok=True)
    output_path = os.path.join("Data2_png", "Data2_while_churn.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()