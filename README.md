<h1>분류 문제 머신러닝 앱</h1> <br>
<br>
<h2>[기능]</h2>
- 데이터 업로드<br>
- 간단한 EDA<br>
- 결측치, 이상치 처리(대체, 삭제)<br>
- 데이터 스케일링(정규화, 표준화) / 엔코딩(라벨, 원핫)<br>
- 클래스 불균형(over, under sampling)<br>
- 모델링(Logistic Regression, Decision Tree, Random Forest)<br>
<br>
<h2>[설명]</h2>
전처리 과정이 4단계로 구성되어있음<br>
- STEP1 : 데이터의 info, describe, value_counts 확인, 타겟 변수 설정, 각 변수의 그래프를 확인, 필요없는 변수 삭제<br>
- STEP2 : 데이터의 결측치 및 이상치 유무 확인 후 삭제 혹은 대체할 지 결정<br>
- STEP3 : 수치형 변수가 있다면 데이터 스케일링(정규화, 표준화 선택) / 범주형 변수가 있다면 데이터 엔코딩(라벨, 원핫 선택) (*y_train은 라벨엔코딩)<br>
- STEP4 : 클래스의 비율 확인 후 over sampling / under sampling을 할 지 결정 (안해도 됨)<br>
<br>
각 STEP을 순차적으로 해야 함.
<br>
ex. STEP4에서 STEP1로 돌아간다면 처음부터 다시 순차적으로 STEP을 진행해야 함
