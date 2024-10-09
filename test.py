import pandas as pd
import numpy as np

# 예시 데이터프레임
data = {
    'A': [1, 2, np.nan],
    'B': [4, np.nan, 6],
    'C': [5, np.nan, 9],
    'D': [1, 2, 3]
}
df = pd.DataFrame(data)

# 1. 결측치가 있는 행의 인덱스 찾기
missing_indices = df[df.isnull().any(axis=1)].index.tolist()

# 2. 결측치가 있는 행 출력
print("결측치가 있는 행:")
print(df.loc[missing_indices])

# 3. 결측치를 각 열의 평균값으로 대체
df_filled = df.fillna(df.mean())

# 4. 결과 확인
print("\n결측치 처리 후 데이터프레임 (평균값으로 대체):")
print(df_filled.loc[missing_indices])
