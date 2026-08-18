import Data1_LGBM
import read_data as rd
import Data1_churn_prediction as cp
import Data1_reason_for_churn as rc
import Data2_first_purchase_days
import Data2_processing as pp
import Data2_customer_analysis
import Data2_whale_payment
import Data2_whale_genres
import Data2_while_churn
import Data2_churn
###
#總體流程 1.發現把玩家留下才產生付費意願 2.避免玩家流失，玩家為何會流失的檢討 3.流失模型預測
###
#-------------------------------------------------------------
# 0. 讀取資料集
df_a = rd.main('online_gaming_behavior_dataset.csv') # 'online_gaming_behavior_dataset.csv'
#-------------------------------------------------------------
# 1. data1 特徵處理 LGBM_learning and 輸出特徵重要性三張圖
Data1_LGBM.main(df_a)
# 傳統誤區：「因為玩家課金了，所以他會留下來。」（錯誤）
# 數據事實：「因為玩家先喜歡遊戲並留了下來（高遊玩時間），他才產生付費意願。」（正確）
# 關鍵結論：遊玩時間是留存的「先驗指標（Leading Indicator）」，
# 而課金通常是高黏著度帶來的「結果（Lagging Outcome）」。
#-------------------------------------------------------------
# 2.避免玩家流失，玩家為何會流失的檢討
rc.main(df_a)
#-------------------------------------------------------------
# 3.玩家流失模型預測
cp.main(df_a)
# 預測玩家流失的商業價值
# 流失模型能幫你精準標記出「即將流失的高風險群體」（如流失機率 > 70%）。
# 團隊只需將預算專注投放在這 10%~15% 的高風險玩家身上，大幅降低行銷與獎勵發放成本。
# 阻止「無效召回」：
# 玩家一旦真正卸載遊戲並流失超過一個月，召回率極低；
# 而透過模型在玩家「開始表現出厭倦/流失前兆」時進行即時干預，挽回成功率比事後召回高出數倍。
#-------------------------------------------------------------
# Data2
print("\nData2\n")
#-------------------------------------------------------------
# 0. 讀取資料集
df_b = rd.main("mobile_game_inapp_purchases.csv") # mobile_game_inapp_purchases
#-------------------------------------------------------------
# 0. 缺值資料處理
df_b_after_processing = pp.main(df_b)
#-------------------------------------------------------------
# 縮短付費轉化週期（First Purchase Days)
Data2_first_purchase_days.main(df_b_after_processing)
#-------------------------------------------------------------
# 遊戲品類（GameGenre）商業化模型優化 x
#-------------------------------------------------------------
# 玩家人數 vs 營收貢獻佔比（二八法則）+ 各客群平均消費（ARPU）
# 精準客群分群與鯨魚玩家（Whale）VIP 運營
Data2_customer_analysis.main(df_b_after_processing)
#-------------------------------------------------------------
# 金流管道（Payment Method）與支付摩擦力優化
# 鯨魚玩家支付管道分佈圓餅圖
Data2_whale_payment.main(df_b_after_processing)
#-------------------------------------------------------------
# 跨國買量（UA）與人口統計精準投放 todo?
#-------------------------------------------------------------

#=============================================================
# TODO
# *鯨魚玩家（Whale）預測
Data2_while_churn.main(df_b_after_processing)
# *流失模型
Data2_churn.main(df_b_after_processing)
# *高風險 高投資 玩家投出禮包
#=============================================================

# 鯨魚玩家最偏好的遊戲品類排行（TOP 6 營收與人數） 遊戲分群(目前沒打算加入)
# Data2_whale_genres.main(df_b_after_processing)