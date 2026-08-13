import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(df1):
    df = df1.copy()
    level_threshold(df)
    game_difficulty(df)
    habit(df)

    plot_eda_summary(df)

def level_threshold(df1): # 門檻瓶頸分析：哪一個等級區間的玩家最容易流失？
    df =df1.copy()
    # 將 PlayerLevel 劃分為等級區間
    df['Level_Group'] = pd.cut(df['PlayerLevel'], bins=[0, 10, 25, 50, 75, 100],
                               labels=['1-10', '11-25', '26-50', '51-75', '76-100'])

    # 計算每個等級區間的「低黏著度/流失率 (%)」
    level_churn = df.groupby('Level_Group')['EngagementLevel'].apply(lambda x: (x == 'Low').mean() * 100)
    print("=== 各等級區間流失率 (%) ===")
    print(round(level_churn, 2))

def game_difficulty(df1):
    df = df1.copy()
    # 計算不同難度的流失率 (%)
    diff_churn = df.groupby('GameDifficulty')['EngagementLevel'].apply(lambda x: (x == 'Low').mean() * 100)
    print("=== 不同難度流失率 (%) ===")
    print(round(diff_churn, 2))

def habit(df1):
    df = df1.copy()
    # 比較每週登入次數 (SessionsPerWeek) 與平均單次時數 (AvgSessionDurationMinutes) 在 Low / High 玩家之間的差異
    habits_summary = df.groupby('EngagementLevel')[
        ['SessionsPerWeek', 'AvgSessionDurationMinutes', 'PlayTimeHours']].mean()
    print("=== 不同黏著度玩家的平均行為特性 ===")
    print(round(habits_summary, 2))


def plot_eda_summary(df1):
    df = df1.copy()

    # 設定圖表整體美觀主題
    sns.set_theme(style="whitegrid")

    # 準備 3 大圖表數據
    df['Level_Group'] = pd.cut(df['PlayerLevel'], bins=[0, 10, 25, 50, 75, 100],
                               labels=['1-10', '11-25', '26-50', '51-75', '76-100'])
    level_churn = df.groupby('Level_Group', observed=False)['EngagementLevel'].apply(
        lambda x: (x == 'Low').mean() * 100)
    diff_churn = df.groupby('GameDifficulty')['EngagementLevel'].apply(lambda x: (x == 'Low').mean() * 100)

    habits = df.groupby('EngagementLevel')[['SessionsPerWeek']].mean().reindex(['Low', 'Medium', 'High'])

    # 建立 1x3 的橫向畫布
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Game Behavioral Analysis & Churn Drivers (EDA Summary)', fontsize=15, fontweight='bold', y=1.02)

    # -----------------------------------------------------------------
    # 圖 1：等級區間流失率
    # -----------------------------------------------------------------
    ax1 = axes[0]
    bars1 = ax1.bar(level_churn.index, level_churn.values, color='#2b5c8f', width=0.55)
    ax1.set_title('1. Churn Rate (%) by Player Level Group', fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel('Player Level Group', fontsize=10)
    ax1.set_ylabel('Low Engagement / Churn Rate (%)', fontsize=10)
    ax1.set_ylim(0, 40)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom',
                 fontweight='bold')

    # -----------------------------------------------------------------
    # 圖 2：遊戲難度流失率
    # -----------------------------------------------------------------
    ax2 = axes[1]
    bars2 = ax2.bar(diff_churn.index, diff_churn.values, color='#d95f02', width=0.45)
    ax2.set_title('2. Churn Rate (%) by Game Difficulty', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel('Game Difficulty', fontsize=10)
    ax2.set_ylabel('Low Engagement / Churn Rate (%)', fontsize=10)
    ax2.set_ylim(0, 40)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom',
                 fontweight='bold')

    # -----------------------------------------------------------------
    # 圖 3：每週登入頻率
    # -----------------------------------------------------------------
    ax3 = axes[2]
    bars3 = ax3.bar(habits.index, habits['SessionsPerWeek'], color=['#d95f02', '#7570b3', '#1b9e77'], width=0.5)
    ax3.set_title('3. Weekly Login Sessions by Engagement Level', fontsize=11, fontweight='bold', pad=10)
    ax3.set_xlabel('Engagement Level', fontsize=10)
    ax3.set_ylabel('Sessions per Week (Mean)', fontsize=10)
    ax3.set_ylim(0, 18)

    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f'{yval:.2f}', ha='center', va='bottom',
                 fontweight='bold')

    plt.tight_layout()
    plt.show()