
def main(df1):
    df = df1.copy()
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
    return df