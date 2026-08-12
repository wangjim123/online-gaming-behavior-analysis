import LGBM
import read_data as rd

# 1. 讀取資料集
df = rd.main()
print(df.shape, df.head(5))
LGBM.main(df)
# 2. 特徵與目標變數處理

