<h1>분류 문제 머신러닝 앱</h1> 

<h2>[기능]</h2>

- 데이터 업로드

- EDA
  
- 결측치, 이상치 처리(대체, 삭제)
  
- 데이터 스케일링(정규화, 표준화) / 엔코딩(라벨, 원핫)
  
- 클래스 불균형(over, under sampling)

- 모델링(Logistic Regression, Decision Tree, Random Forest)

<h2>[설명]</h2>
전처리 과정이 4단계로 구성되어있음

* STEP1
    - 데이터의 info, describe, value_counts 확인, 타겟 변수 설정, 각 변수의 그래프를 확인, 필요없는 변수 삭제
    - 엔코딩이 이미 된 의심 변수 중에서 사용자가 직접 변수를 선택하여 스케일링 시 제거 가능하게 함
    
* STEP2
    - 데이터의 결측치 및 이상치 유무 확인 후 삭제 혹은 대체할 지 결정

* STEP3
    - 수치형 변수가 있다면 데이터 스케일링(정규화, 표준화 선택) / 범주형 변수가 있다면 데이터 엔코딩(라벨, 원핫 선택)
    - 다음 단계에서 클래스 불균형을 확인하고 해결하기 위해 y_train은 라벨엔코딩을 함

* STEP4
    - 클래스의 비율 확인 후 over sampling / under sampling을 할 지 결정 (안해도 됨)
    
* MODELING
    - 모델 선택 및 하이퍼파라미터 설정 후 학습
    - 학습이 종료되면 acc_score와 confusion matrix가 나타남.
        + Logistic Regression (max_iters: 1 ~ 1000)
        + Decision Tree (Max depth: 1 ~ 1000)
        + Random Forest (Max depth: 1 ~ 1000 / n_estimators: 1 ~ 1000)

각 STEP을 순차적으로 해야 함.
ex. STEP4에서 STEP1로 돌아간다면 처음부터 다시 순차적으로 STEP을 진행해야 함

<h2>[주의사항]</h2>
트리 계열 모델에서는 숫자의 크고 작음의 영향이 없으므로 X 데이터셋에 라벨엔코딩이 된 변수가 있어도 상관없지만, 다른 몇 가지 모델에서는 민감하므로, 라벨엔코딩을 추천하지 않음.