import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def main(df1):
    df = df1.copy()

    #-----------------------------------------------------------
    # 真實付費人數  未課金/無金額紀錄玩家數
    # 2. 定義真實付費玩家標籤 (金額不為空且 > 0)
    df["IsPayer"] = df["InAppPurchaseAmount"].notnull() & (
            df["InAppPurchaseAmount"] > 0
    )

    # 基礎數據統計
    total_users = len(df)
    paying_users = df["IsPayer"].sum()
    non_paying_users = total_users - paying_users
    overall_conversion_rate = paying_users / total_users

    print("=== 📊 修正前的資料分析 ===")
    print(f"總玩家數: {total_users}")
    print(
        f"真實付費玩家數 (InAppPurchaseAmount > 0): {paying_users} (付費率:"
        f" {overall_conversion_rate:.2%})"
    )
    print(
        f"未課金 / 無金額紀錄玩家數: {non_paying_users}"
        f" ({non_paying_users / total_users:.2%})\n"
    )

    #-----------------------------------------------------------
    # 資料缺值處理
    # 1. 玩家人口統計特徵處理
    df['Age'] = df['Age'].fillna(df['Age'].median())
    cat_cols = ['Gender', 'Country', 'Device', 'GameGenre']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')

    # 金額缺失處理：按 SpendingSegment 群組中位數精準填補
    df['InAppPurchaseAmount'] = df['InAppPurchaseAmount'].fillna(
        df.groupby('SpendingSegment')['InAppPurchaseAmount'].transform('median')
    )
    # 3其它文字/天數缺失處理
    df['FirstPurchaseDaysAfterInstall'] = df[
        'FirstPurchaseDaysAfterInstall'
    ].fillna(-1)
    df['PaymentMethod'] = df['PaymentMethod'].fillna('Unrecorded')
    df['LastPurchaseDate'] = df['LastPurchaseDate'].fillna('Unrecorded')

    print('✅ 資料處理完成！已精準依據 Whale/Dolphin/Minnow 層級補全消費金額。')

    # ----------------------------------------------------------
    # 定義付費標記 (1: 有課金, 0: 未課金)
    df["IsPayer"] = df["InAppPurchaseAmount"].notnull() & (
            df["InAppPurchaseAmount"] > 0
    )

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
    # ----------------------------------------------------
    # 留存活躍度 (SessionCount) vs 付費轉化率比較


    # ----------------------------------------------------
