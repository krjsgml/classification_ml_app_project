import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import Qt
from io import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import threading


class ML_app(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("ML Classification Application")
        self.setWindowIcon(QIcon('project_1/ml_app/icon_img/app_icon.png'))
        self.resize(700, 500)
        self.center()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # QSplitter 추가
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        self.leftlayout = QVBoxLayout()
        self.rightlayout = QVBoxLayout()

        # QSplitter에 레이아웃 추가
        left_widget = QWidget()
        left_widget.setLayout(self.leftlayout)
        self.splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_widget.setLayout(self.rightlayout)
        self.splitter.addWidget(right_widget)

        self.splitter.setSizes([700, 300])  # 원하는 비율에 따라 크기를 설정

        self.model_setup_layout = QVBoxLayout()
        self.model_hyperparameter_layout = QVBoxLayout()
        self.preprocess_layout = QVBoxLayout()
        self.preprocess_step_layout = QVBoxLayout()

        self.leftlayout.addLayout(self.model_setup_layout)
        self.leftlayout.addLayout(self.model_hyperparameter_layout)
        self.leftlayout.addLayout(self.preprocess_layout)
        self.rightlayout.addLayout(self.preprocess_step_layout)

        # Main Image
        self.image_label = QLabel(self)
        pixmap = QPixmap('project_1/ml_app/icon_img/start_page.png')
        pixmap = pixmap.scaled(700, 500, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.main_layout.addWidget(self.image_label)

        # menu bar
        self.create_menu()

    def create_menu(self):
        # menu바에서 데이터셋 로드
        menubar = self.menuBar()
        data_menu = menubar.addMenu('&Data')
        self.load_data_menu(data_menu)

    def load_data_menu(self, menu):
        # 데이터셋 폴더 안에 있는 데이터셋 목록들을 메뉴로
        data_folder = 'project_1/ml_app/data'
        if os.path.exists(data_folder):
            for filename in os.listdir(data_folder):
                file_action = QAction(filename, self)
                file_action.triggered.connect(lambda _, fname=filename: self.load_data(fname))
                menu.addAction(file_action)
        else:
            print(f"Folder '{data_folder}' not found.")

    def load_data(self, filename):
        data_path = os.path.join('project_1/ml_app/data', filename)
        if os.path.exists(data_path):
            self.selected_data = pd.read_csv(data_path)     # 선택된 파일을 dataframe으로
            QMessageBox.about(self, 'Load Data', f'Data loaded: {filename}')

            # 데이터가 새로 로드되었을 시에 모든 layout clear
            layouts = [self.model_setup_layout, self.model_hyperparameter_layout, self.preprocess_layout, self.preprocess_step_layout]
            for layout in layouts:
                self.clear_layout(layout)

            self.image_label.hide()     # 메인이미지 숨기기
            self.clear_layout
            self.preprocess()           # 전처리 단계 시작

            print(f"Selected Data head\n{self.selected_data.head()}")
        
    def preprocess(self):
        preprocessing_layout = QVBoxLayout()
        self.step1 = QPushButton('STEP1')
        self.step2 = QPushButton('STEP2')
        self.step3 = QPushButton('STEP3')
        self.step4 = QPushButton('STEP4')

        preprocessing_layout.addWidget(self.step1)
        preprocessing_layout.addWidget(self.step2)
        preprocessing_layout.addWidget(self.step3)
        preprocessing_layout.addWidget(self.step4)

        self.step1.clicked.connect(self.EDA)
        self.step2.clicked.connect(self.missing_outlier_value)
        self.step3.clicked.connect(self.data_scale_encoding)
        self.step4.clicked.connect(self.class_imbalance)

        # 각각의 step은 전 단계를 수행해야 setEnabled(True)로 바뀜
        # (현재 단계를 수행해야 다음 단계로 넘어갈 수 있음)
        self.step2.setEnabled(False)
        self.step3.setEnabled(False)
        self.step4.setEnabled(False)

        self.preprocess_step_layout.addLayout(preprocessing_layout)

    def EDA(self):
        self.clear_layout(self.preprocess_layout)
        self.clear_layout(self.model_setup_layout)
        self.clear_layout(self.model_hyperparameter_layout)
        print("=================STEP1=================")

        # 각 step별 개별적인 데이터셋이 필요
        self.step1_data = self.selected_data.copy()

        self.step2.setEnabled(False)
        self.step3.setEnabled(False)
        self.step4.setEnabled(False)

        EDA_layout = QVBoxLayout()
        EDA_menu_layout = QHBoxLayout()
        EDA_show_layout = QVBoxLayout()
        
        # 타겟변수 설정
        select_target_variable_layout = QHBoxLayout()
        select_target_variable_layout.addWidget(QLabel("Select Target Variable :"))

        self.target_variable_selection = QComboBox(self)
        select_target_variable_layout.addWidget(self.target_variable_selection)

        self.target_variable_selection.addItem('Select Variable')
        self.target_variable_selection.addItems(self.step1_data)
        self.target_variable_selection.activated[str].connect(self.target_variable_split)

        # 변수 그래프 보여주기
        select_show_graph_variable_layout = QHBoxLayout()
        select_show_graph_variable_layout.addWidget(QLabel("Select Show Graph :"))

        self.graph_variable_selection = QComboBox(self)
        select_show_graph_variable_layout.addWidget(self.graph_variable_selection)

        self.graph_variable_selection.addItem("Select Variable")
        self.graph_variable_selection.addItems(self.step1_data)

        self.show_graph_btn = QPushButton("Show")
        select_show_graph_variable_layout.addWidget(self.show_graph_btn)
        self.show_graph_btn.clicked.connect(self.show_graph)

        # 변수 삭제
        select_drop_variable_layout = QHBoxLayout()
        select_drop_variable_layout.addWidget(QLabel("Select Drop Variable :"))
        
        self.drop_variable_selection = QComboBox(self)
        select_drop_variable_layout.addWidget(self.drop_variable_selection)

        self.drop_variable_selection.addItem("Select Variable")
        self.drop_variable_selection.addItems(self.step1_data.columns)
        
        drop_btn = QPushButton("Drop")
        select_drop_variable_layout.addWidget(drop_btn)
        drop_btn.clicked.connect(self.drop_ok)

        EDA_layout.addLayout(EDA_menu_layout, 1)
        EDA_layout.addLayout(select_target_variable_layout, 1)
        EDA_layout.addLayout(select_show_graph_variable_layout, 1)
        EDA_layout.addLayout(select_drop_variable_layout, 1)
        EDA_layout.addStretch(1)
        EDA_layout.addLayout(EDA_show_layout, 5)

        self.preprocess_layout.addLayout(EDA_layout)
        self.preprocess_layout.addStretch(3)

        info = QPushButton('info')
        describe = QPushButton('describe')
        value_counts = QPushButton('value counts')

        EDA_menu_layout.addWidget(info)
        EDA_menu_layout.addWidget(describe)
        EDA_menu_layout.addWidget(value_counts)

        # info()는 출력하고 none을 반환하기 때문에 아래 과정이 필요
        buffer = StringIO()
        self.step1_data.info(buf=buffer)
        data_info = buffer.getvalue()

        # describe
        data_describe = self.step1_data.describe()

        # info 버튼 클릭 시 info() 메서드를 이용해 앱에 출력
        info.clicked.connect(lambda : self.clear_layout(EDA_show_layout))
        info.clicked.connect(lambda : display_info(EDA_show_layout, data_info))

        # describe 버튼 클릭 시 describe() 메서드를 이용해 앱에 출력
        describe.clicked.connect(lambda : self.clear_layout(EDA_show_layout))
        describe.clicked.connect(lambda : display_describe(EDA_show_layout, data_describe))

        # value_counts 버튼 클릭 시 value_counts() 메서드를 이용해 앱에 출력
        value_counts.clicked.connect(lambda : self.clear_layout(EDA_show_layout))
        value_counts.clicked.connect(lambda : select_value_counts(EDA_show_layout, self.step1_data))

        def display_info(layout, info):
            # info 버튼을 클릭 시 고정폭 폰트를 이용해 가독성 향상
            text_edit = QPlainTextEdit()
            text_edit.setPlainText(info)
            text_edit.setReadOnly(True)
            
            font = QFont("Courier New")
            font.setStyleHint(QFont.Monospace)
            text_edit.setFont(font)
            
            layout.addWidget(text_edit)

        def display_describe(layout, df):
            # QTableWidget을 사용해 가독성 향상
            table = QTableWidget()
            table.setRowCount(len(df))
            table.setColumnCount(len(df.columns) + 1)

            # 인덱스 열 헤더를 추가 (통계 값 이름: mean, max, min 등)
            headers = [''] + list(df.columns)
            table.setHorizontalHeaderLabels(headers)

            # 데이터 삽입
            for row in range(len(df)):
                # 인덱스 값을 첫 번째 열에 삽입
                table.setItem(row, 0, QTableWidgetItem(str(df.index[row])))

                for col in range(len(df.columns)):
                    table.setItem(row, col + 1, QTableWidgetItem(str(df.iloc[row, col])))

            layout.addWidget(table)

        def display_value_counts(layout, df, col):
            # QTableWidget을 사용해 가독성 향상
            # display_describe()와 구조는 똑같음. (다만, describe에서는 앞에 통계 값의 이름들을 추가해야하므로 추가 과정이 필요하였음.)
            # 추후에 display_describe()와 display_value_counts()를 하나의 메서드로 구현할 예정
            if col:
                self.clear_layout(layout)

                df = df[col].value_counts().reset_index()   # df[col].value_counts()는 시리즈이므로 데이터프레임 형식으로 맞춰줘서 Table로
                table = QTableWidget()
                table.setRowCount(len(df))
                table.setColumnCount(len(df.columns))
                table.setHorizontalHeaderLabels(df.columns)

                # 데이터 삽입
                for row in range(len(df)):
                    for col in range(len(df.columns)):
                        table.setItem(row, col, QTableWidgetItem(str(df.iloc[row, col])))

                layout.addWidget(table)

        def select_value_counts(layout, df):
            # 각각의 변수들 하나하나 value_counts()를 확인하고자 QCombox를 만들어 변수를 선택하면
            # display_value_counts()가 호출되어 선택한 변수의 value_counts()를 확인할 수 있음.
            select_col_layout = QVBoxLayout()
            show_value_counts_layout = QHBoxLayout()

            select_col_value_counts = QComboBox()
            select_col_layout.addWidget(select_col_value_counts)

            layout.addLayout(select_col_layout)
            layout.addLayout(show_value_counts_layout)

            select_col_value_counts.addItems(df.columns)
            select_col_value_counts.currentTextChanged.connect(lambda col : display_value_counts(show_value_counts_layout, df, col))

    def show_graph(self):
        # Qcombobox에서 선택한 변수의 그래프를 출력
        variable = self.graph_variable_selection.currentText()
        if variable != "Select Variable":
            print(f"Show Graph Variable : {variable}")
            if self.step1_data[variable].dtype in ['int', 'float']: # 수치형은 histplot 그래프
                sns.histplot(data=self.step1_data, x=self.step1_data[variable])

            else:   # 범주형은 bar 그래프
                self.step1_data[variable].value_counts().plot(kind='bar', alpha=0.7)

            plt.show()

    def drop_ok(self):
        # 삭제할 변수를 선택하면 삭제시켜줌.
        variable = self.drop_variable_selection.currentText()
        if variable != "Select Variable":
            print(f"Drop Variable : {variable}")
            self.step1_data.drop(columns=[variable], inplace=True)

            # 타겟변수 선택, 그래프로 표시할 변수 선택, 삭제할 변수 선택 QCombobox에서 해당 변수 삭제
            target_col_index = self.target_variable_selection.findText(variable)
            graph_col_index = self.graph_variable_selection.findText(variable)
            drop_col_index = self.drop_variable_selection.findText(variable)

            self.target_variable_selection.removeItem(target_col_index)
            self.graph_variable_selection.removeItem(graph_col_index)
            self.drop_variable_selection.removeItem(drop_col_index)
            
            QMessageBox.about(self, "Delete Variable", f"drop {variable}")

    def target_variable_split(self, text):
        # 타겟 변수 선택 시 self.target_variable로 저장
        # x,y 분리는 이상치와 결측치 제거 후 진행
        if text != 'Select Variable':
            print(f"Target Variable : {text}")
            self.target_variable = text

            # 다음 단계 (step2) 이동 가능
            self.step2.setEnabled(True)


    def missing_outlier_value(self):
        self.clear_layout(self.preprocess_layout)
        self.clear_layout(self.model_setup_layout)
        self.clear_layout(self.model_hyperparameter_layout)
        print("\n=================STEP2=================")

        self.step3.setEnabled(False)
        self.step4.setEnabled(False)

        self.step2_data = self.step1_data.copy()

        # 결측치나 이상치를 찾았을 때 1로 바뀜
        self.missing_found = 0
        self.outlier_found = 0

        missing_layout = QVBoxLayout()
        outlier_layout = QVBoxLayout()

        self.preprocess_layout.addLayout(missing_layout)
        self.preprocess_layout.addLayout(outlier_layout)

        self.missing_outlier_ok = QPushButton("OK")

        self.preprocess_layout.addStretch(1)
        self.preprocess_layout.addWidget(self.missing_outlier_ok)
        self.preprocess_layout.addStretch(2)

        missing_cols = []   # 결측치가 있는 변수 list
        for col in self.step2_data:
            if self.step2_data[col].isnull().any():
                missing_cols.append(col)

        print(f"Find col have missing value {str(missing_cols)}")

        # 결측치 발견
        if len(missing_cols) > 0:
            self.missing_found = 1

        if self.missing_found:
            # 삭제 혹은 대체 선택
            missing_layout.addWidget(QLabel('Select missing value Delete or Replace'))
            select_missing_value = QHBoxLayout()
            self.missing_val_delete = QCheckBox("Delete")
            self.missing_val_replace = QCheckBox("Replace")

            select_missing_value.addWidget(self.missing_val_delete)
            select_missing_value.addWidget(self.missing_val_replace)

            self.missing_val_delete.toggled.connect(lambda: self.missing_val_replace.setChecked(False))
            self.missing_val_replace.toggled.connect(lambda: self.missing_val_delete.setChecked(False))
            missing_layout.addLayout(select_missing_value)

        else:
            missing_layout.addWidget(QLabel('Not found missing values'))

        outlier_cols = []   # 이상치가 있는 변수 list
        for col in self.step2_data.select_dtypes(include=['float', 'int']):
            lower, upper = self.outlier_bound(col)
            if ((self.step2_data[col] < lower) | (self.step2_data[col] > upper)).any():
                outlier_cols.append(col)
                self.outlier_found = 1

        print(f"Find col have outlier value {str(outlier_cols)}")

        # 이상치 발견
        if self.outlier_found:
            # 삭제 혹은 대체 선택
            outlier_layout.addWidget(QLabel('select outlier value Delete or Replace'))
            select_outlier_value = QHBoxLayout()
            self.outlier_val_delete = QCheckBox("Delete")
            self.outlier_val_replace = QCheckBox("Replcae")

            select_outlier_value.addWidget(self.outlier_val_delete)
            select_outlier_value.addWidget(self.outlier_val_replace)

            self.outlier_val_delete.toggled.connect(lambda: self.outlier_val_replace.setChecked(False))
            self.outlier_val_replace.toggled.connect(lambda: self.outlier_val_delete.setChecked(False))
            outlier_layout.addLayout(select_outlier_value)

        else:
            outlier_layout.addWidget(QLabel('Not found outlier values'))

        self.missing_outlier_ok.clicked.connect(self.missing_outlier_preprocess)

    def missing_outlier_preprocess(self):
        # 결측치는 무조건 처리시켜줘야하지만, 이상치는 처리를 안시켜줘도 됨.
        error = 0   # 결측치 체크박스에 체크가 안되어있을 경우에 1로 됨

        if self.missing_found:
            # 삭제 체크박스에 체크된 경우
            if self.missing_val_delete.isChecked():
                self.step2_data.dropna(inplace=True)

            # 대체 체크박스에 체크된 경우
            elif self.missing_val_replace.isChecked():
                # 결측치부분이 어떻게 대체되었는지 확인하기 위해 missing_indices에 결측치가 있는 행 인덱스 저장
                missing_indices = self.step2_data[self.step2_data.isnull().any(axis=1)].index.tolist()

                nums = self.step2_data.select_dtypes(include=['int', 'float']).columns.tolist()
                objs = self.step2_data.select_dtypes(include=['object']).columns.tolist()

                print(f"\n결측치 대체 전 해당 행\n{self.step2_data.iloc[missing_indices]}")

                # 수치형은 mean()으로 대체
                for num in nums:
                    self.step2_data[num] = self.step2_data[num].fillna(self.step2_data[num].mean())
                # 범주형은 mode()로 대체
                for obj in objs:
                    self.step2_data[obj] = self.step2_data[obj].fillna(self.step2_data[obj].mode().iloc[0])
                
                print(f"\n결측치 대체 후 해당 행\n{self.step2_data.iloc[missing_indices]}")
                print(f"결측치 갯수 : {self.step2_data.isnull().sum()}")

            else:
                error = 1

        if self.outlier_found:
            # 삭제 체크박스에 체크된 경우
            if self.outlier_val_delete.isChecked():
                for col in self.step2_data.select_dtypes(include=['float', 'int']).columns:
                    # 분류 문제에 타겟 변수는 보통 범주형 변수(수치형 변수라면 이미 엔코딩된 데이터)이므로 제외
                    if col != self.target_variable:
                        lower, upper = self.outlier_bound(col)
                        # lower보다 낮으면, upper보다 높으면 해당 행 삭제 (lower <= data <= upper로만 저장)
                        self.step2_data = self.step2_data[(self.step2_data[col] >= lower) & (self.step2_data[col] <= upper)]

            # 대체 체크박스에 체크된 경우
            elif self.outlier_val_replace.isChecked():
                 for col in self.step2_data.select_dtypes(include=['float', 'int']).columns:
                    # 분류 문제에 타겟 변수는 보통 범주형 변수(수치형 변수라면 이미 엔코딩된 데이터)이므로 제외
                    if col != self.target_variable:
                        lower, upper = self.outlier_bound(col)
                        # lower보다 낮으면 col_min, upper보다 높으면 col_max로 대체
                        col_min = self.step2_data[col].min()
                        col_max = self.step2_data[col].max()

                        self.step2_data[col] = self.step2_data[col].apply(
                            lambda x: col_min if x < lower else (col_max if x > upper else x)
                        )
        
        if error == 1:
            QMessageBox.about(self, "erorr", "select please")

        else:
            self.clear_layout(self.preprocess_layout)

            print(f"Selected Data Head\n{self.step2_data.head()}")
            print(f"\ny_value_counts\n{self.step2_data[self.target_variable].value_counts()}")

            self.preprocess_layout.addWidget(QLabel("Go Next Step"))
            QMessageBox.about(self, "complete", "missing val & outlier val preprocess complete")

            self.step3.setEnabled(True)

    def outlier_bound(self, col):
        # IQR을 이용하여 lower와 upper를 반환하는 함수
        Q1 = self.step2_data[col].quantile(0.25)
        Q3 = self.step2_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return lower, upper
    
    def data_scale_encoding(self):
        # train-test 데이터 비율 설정 / 데이터 스케일링 / 엔코딩
        self.clear_layout(self.preprocess_layout)
        self.clear_layout(self.model_setup_layout)
        self.clear_layout(self.model_hyperparameter_layout)
        print("\n=================STEP3=================")

        self.step4.setEnabled(False)

        self.step3_data = self.step2_data.copy()

        # set train-test-split ratio part
        self.preprocess_layout.addWidget(QLabel("train-test-split ratio"))
        self.train_test_split_input = QSpinBox()
        self.preprocess_layout.addWidget(self.train_test_split_input)
        self.train_test_split_input.setMaximum(100)
        self.train_test_split_input.setMinimum(0)
        self.train_test_split_input.setValue(70)

        self.X = self.step3_data.drop(columns=[self.target_variable])
        self.y = self.step3_data[self.target_variable]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=float(self.train_test_split_input.value()/100), random_state=42, stratify=self.y
        )
        print(f"train_test_split_ratio(%): {self.train_test_split_input.value()}")
        print(f"\nX_train head\n{self.X_train.head()}")
        print(f"\nX_test head\n{self.X_test.head()}")
        print(f"\ny_train head\n{self.y_train.head()}")
        print(f"\ny_train value counts\n{self.y_train.value_counts()}")
        print(f"\ny_test head\n{self.y_test.head()}")
        print(f"\ny_test value counts\n{self.y_test.value_counts()}")

        self.train_test_split_input.valueChanged.connect(self.train_test_split_)

        # data scaling part
        data_scale_layout = QHBoxLayout()
        self.preprocess_layout.addLayout(data_scale_layout)
        self.nums_X_train = list(self.X_train.select_dtypes(include=['float', 'int']).columns)
        self.nums_X_test = list(self.X_test.select_dtypes(include=['float', 'int']).columns)

        # data encoding part
        data_encoding_layout = QHBoxLayout()
        self.preprocess_layout.addLayout(data_encoding_layout)
        self.objs_X_train = list(self.X_train.select_dtypes(include=['object']).columns)
        self.objs_X_test = list(self.X_test.select_dtypes(include=['object']).columns)

        self.scale_encoding_ok = QPushButton("OK")
        self.preprocess_layout.addStretch(2)
        self.preprocess_layout.addWidget(self.scale_encoding_ok)
        self.scale_encoding_ok.clicked.connect(self.scale_encoding_preprocess)

        if len(self.nums_X_train) > 0:
            # 수치형 변수가 있다면
            select_scale = QVBoxLayout()
            # 정규화 / 표준화 선택
            self.normalization = QCheckBox("normalization")
            self.standardization = QCheckBox("standardization")
            select_scale.addWidget(self.normalization)
            select_scale.addWidget(self.standardization)
            data_scale_layout.addWidget(QLabel("select to data scale method"))
            data_scale_layout.addLayout(select_scale)

            self.normalization.toggled.connect(lambda: self.standardization.setChecked(False))
            self.standardization.toggled.connect(lambda: self.normalization.setChecked(False))

        else:
            data_scale_layout.addWidget(QLabel("No need to scale"))

        if len(self.objs_X_train) > 0:
            # 범주형 변수가 있다면
            select_encoding = QVBoxLayout()
            # label encoding / one-hot encoding 선택
            self.label_encoding = QCheckBox("Label Encoding")
            self.one_hot_encoding = QCheckBox("One-Hot Encoding")
            select_encoding.addWidget(self.label_encoding)
            select_encoding.addWidget(self.one_hot_encoding)
            data_encoding_layout.addWidget(QLabel("select to data encoding method"))
            data_encoding_layout.addLayout(select_encoding)

            self.label_encoding.toggled.connect(lambda: self.one_hot_encoding.setChecked(False))
            self.one_hot_encoding.toggled.connect(lambda: self.label_encoding.setChecked(False))

        else:
            data_encoding_layout.addWidget(QLabel("No need to encoding"))

    def scale_encoding_preprocess(self):
        # scaling 대상: X_train, X_test의 수치형 변수들
        # encoding 대상: X_train, X_test의 범주형 변수들
        # y_train은 후에 클래스 불균형 문제를 해소하기 위해 label encoding을 해야함.
        # y_test는 후에 y_train의 라벨엔코딩을 역변환하여 비교하면 되어 encoding 생략
        # 혹은 이미 y(타겟 변수)가 엔코딩이 되어있을 수도 있음

        # 각각의 과정은 해당 변수들을 출력해줌.(ex. scaling 대상이 되는 변수들을 출력)
        # 또한 각각의 과정 후의 데이터들을 head()를 이용해 출력

        error = 0   # 체크박스에 체크가 안되어있다면 1이 됨

        if len(self.nums_X_train) > 0:
            if self.normalization.isChecked():
                # 정규화 선택 시
                print(f"data scaling(Normalization) : {self.nums_X_train}")
                scaler = MinMaxScaler()
                self.X_train[self.nums_X_train] = scaler.fit_transform(self.X_train[self.nums_X_train])
                self.X_test[self.nums_X_test] = scaler.transform(self.X_test[self.nums_X_test])

                print(f"\nAfter Scaling X_train head\n{self.X_train.head()}")
                print(f"\nAfter scaling X_test head\n{self.X_test.head()}")
            
            elif self.standardization.isChecked():
                # 표준화 선택 시
                print(f"data scaling(Standardization) : {self.nums_X_train}")
                scaler = StandardScaler()
                self.X_train[self.nums_X_train] = scaler.fit_transform(self.X_train[self.nums_X_train])
                self.X_test[self.nums_X_test] = scaler.transform(self.X_test[self.nums_X_test])

                print(f"\nAfter Scaling X_train head\n{self.X_train.head()}")
                print(f"\nAfter scaling X_test head\n{self.X_test.head()}")

            else:
                error = 1

        if len(self.objs_X_train) > 0:
            if self.label_encoding.isChecked():
                # 라벨 엔코딩 선택 시
                print(f"encoding(Label Encoding) : {self.objs_X_train}")
                encoder = LabelEncoder()
                for col in self.objs_X_train:
                    try:
                        self.X_train[col] = encoder.fit_transform(self.X_train[[col]])
                        self.X_test[col] = encoder.transform(self.X_test[col])
                    except:
                        self.X_train = self.X_train.drop(columns=[col])
                        self.X_test = self.X_test.drop(columns=[col])
                        QMessageBox.about(self, "error", f"drop {col}")

                print(f"\nAfter Encoding X_train head\n{self.X_train.head()}")
                print(f"\nAfter Encoding X_test head\n{self.X_test.head()}")

            elif self.one_hot_encoding.isChecked():
                # 원-핫 엔코딩 선택 시
                print(f"encoding(One-Hot Encoding) : {self.objs_X_train}")
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                for col in self.objs_X_train:
                    transformed_train = encoder.fit_transform(self.X_train[[col]])
                    encoded_cols = [f"{col}_{category}" for category in encoder.categories_[0]]
                    transformed_train_df = pd.DataFrame(transformed_train, columns=encoded_cols, index=self.X_train.index)

                    transformed_test = encoder.transform(self.X_test[[col]])
                    transformed_test_df = pd.DataFrame(transformed_test, columns=encoded_cols, index=self.X_test.index)

                    self.X_train = pd.concat([self.X_train, transformed_train_df], axis=1).drop(columns=[col])
                    self.X_test = pd.concat([self.X_test, transformed_test_df], axis=1).drop(columns=[col])

                print(f"\nAfter Encoding X_train head\n{self.X_train.head()}")
                print(f"\nAfter Encoding X_test head\n{self.X_test.head()}")

            else:
                error = 1
        
        self.label_encoder = None
        # 이미 엔코딩이 되어있는 경우를 위해 y_train이 object type이면 엔코딩하기
        if self.y_train.dtype == 'object':
            print(f"encoding(Label Encoding) - y_train dataset : {self.y_train.name}")
            self.label_encoder = LabelEncoder()
            self.y_train = self.label_encoder.fit_transform(self.y_train)
            self.y_train = pd.Series(self.y_train)

            print(f"\nAfter Encoding y_train head\n{self.y_train.head()}")

        if error == 1:
            QMessageBox.about(self, "error", "select please")

        else:
            self.clear_layout(self.preprocess_layout)
            self.preprocess_layout.addWidget(QLabel("Go Next Step"))
            QMessageBox.about(self, "complete", "scaling & encoding complete")

            self.step4.setEnabled(True)

    def train_test_split_(self):
        # train-test-split ratio의 값이 바뀌면 다시 train-test data 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=float(self.train_test_split_input.value()/100), random_state=42, stratify=self.y
        )
        print(f"train_test_split_ratio(%): {self.train_test_split_input.value()}")

    def class_imbalance(self):
        # 해당 단계에 들어서자마자 y_train의 비율을 보여주는 그래프를 출력
        # over_sampling / under_sampling / original 각각의 버튼을 클릭하면 해당 과정 후의 비율을 보여줌
        # ok버튼을 눌러야 해당 과정이 저장됨
        self.clear_layout(self.model_setup_layout)
        self.clear_layout(self.model_hyperparameter_layout)
        self.clear_layout(self.preprocess_layout)

        print("\n=================STEP4=================")
        print(f"\ny_train value counts\n{self.y_train.value_counts()}")

        self.sampling_x = None
        self.sampling_y = None

        class_imbalance_layout = QVBoxLayout()
        self.preprocess_layout.addLayout(class_imbalance_layout)
        class_imbalance_layout.addStretch(1)
        
        class_imbalance_layout.addWidget(QLabel("select over/under sampling"))
        
        class_imbalance_method_layout = QHBoxLayout()
        class_imbalance_layout.addLayout(class_imbalance_method_layout)

        oversampling = QPushButton("OverSampling")
        undersampling = QPushButton("UnderSampling")
        original = QPushButton("Original")
        class_imbalance_method_layout.addWidget(oversampling)
        class_imbalance_method_layout.addWidget(undersampling)
        class_imbalance_method_layout.addWidget(original)
        sampling_ok = QPushButton("OK")
        class_imbalance_method_layout.addWidget(sampling_ok)

        self.plot_target_variable(self.y_train)
        class_imbalance_layout.addStretch(2)

        oversampling.clicked.connect(self.oversampling_process)
        undersampling.clicked.connect(self.undersampling_precess)
        original.clicked.connect(self.original_process)
        sampling_ok.clicked.connect(self.sampling_process_ok)

    def plot_target_variable(self, target):
        # y_train 비율 plot
        class_counts = target.value_counts()

        class_counts.plot(kind='bar')
        plt.title('train - target variable')
        plt.xlabel('Class')
        plt.ylabel('Freq')
        plt.show()

    def oversampling_process(self):
        # oversampling
        self.sampling_x = self.X_train.copy()
        self.sampling_y = self.y_train.copy()
        self.sampling_x, self.sampling_y = RandomOverSampler(random_state=0).fit_resample(self.sampling_x, self.sampling_y)
        self.sampling_y = pd.DataFrame(self.sampling_y, columns=[self.target_variable])
        print(f"\ny_train value counts\n{self.sampling_y.value_counts()}")
       
        self.plot_target_variable(self.sampling_y)

    def undersampling_precess(self):
        # undersampling
        self.sampling_x = self.X_train.copy()
        self.sampling_y = self.y_train.copy()
        self.sampling_x, self.sampling_y = RandomUnderSampler(random_state=0).fit_resample(self.sampling_x, self.sampling_y)
        self.sampling_y = pd.DataFrame(self.sampling_y, columns=[self.target_variable])
        print(f"\ny_train value counts\n{self.sampling_y.value_counts()}")
        self.plot_target_variable(self.sampling_y)

    def original_process(self):
        # original
        if self.sampling_x is not None:
            # 이미 over나 under sampling을 거쳤으면 다시 샘플링데이터를 None으로 저장
            self.sampling_x = None
            self.sampling_y = None
        self.plot_target_variable(self.y_train)

    def sampling_process_ok(self):
        # ok버튼을 누르면 해당 과정 후의 데이터로 저장
        if self.sampling_x is not None:
            # sampling이 되었다면 해당 샘플링 데이터를 훈련데이터로 저장
            self.X_train = self.sampling_x
            self.y_train = self.sampling_y

        self.y_train = self.y_train.squeeze() # squeeze()로 1차원으로 변환
        self.clear_layout(self.preprocess_layout)
        QMessageBox.about(self, "complete", "Finished Preprocessing Step \n Now Go to the Modeling")
        self.model_setup()

    def model_setup(self):
        # model 선택
        # Logistic Regression / Decision Tree / Random Forest
        self.model_selection = QComboBox(self)
        self.model_selection.addItem('model selection')
        self.model_selection.addItem('Logistic Regression')
        self.model_selection.addItem('Decision Tree')
        self.model_selection.addItem('Random Forest')

        model_selection_layout = QHBoxLayout()
        model_selection_layout.addWidget(QLabel("Model Selection : "))
        model_selection_layout.addWidget(self.model_selection)
        self.model_setup_layout.addLayout(model_selection_layout)

        # 모델이 선택된다면 hyperparamer를 설정할 수 있게 해줌
        self.model_selection.activated[str].connect(self.hyperparameter_setup)

    def hyperparameter_setup(self, model_name):
        # 모델의 하이퍼파라미터를 설정
        self.clear_layout(self.model_hyperparameter_layout)
        hyperparameter_layout = QVBoxLayout()
        train_layout = QVBoxLayout()
        self.model_hyperparameter_layout.addLayout(hyperparameter_layout, 7)
        self.model_hyperparameter_layout.addLayout(train_layout, 3)

        train_model = QPushButton("Train Model")
        train_layout.addWidget(train_model)
        train_model.clicked.connect(self.train)

        self.model = None

        if model_name == 'Logistic Regression':
            # Logistic Regression 선택
            # hyperparameter : max_iters
            self.max_iters= QSpinBox()
            self.max_iters.setMinimum(1)
            self.max_iters.setMaximum(1000)
            max_iters_layout = QHBoxLayout()
            max_iters_layout.addWidget(QLabel("Max Iterations : "))
            max_iters_layout.addWidget(self.max_iters)
            hyperparameter_layout.addLayout(max_iters_layout)

            self.model = LogisticRegression(max_iter=self.max_iters.value())
            self.max_iters.valueChanged.connect(self.logistic_model_update)

        elif model_name == 'Decision Tree':
            # Decision Tree 선택
            # hyperparamer : max_depth
            self.dt_max_depth = QSpinBox()
            self.dt_max_depth.setMinimum(1)
            self.dt_max_depth.setMaximum(1000)
            max_depth_layout = QHBoxLayout()
            max_depth_layout.addWidget(QLabel('Max Depth : '))
            max_depth_layout.addWidget(self.dt_max_depth)
            hyperparameter_layout.addLayout(max_depth_layout)

            self.model = DecisionTreeClassifier(max_depth=self.dt_max_depth.value())
            # 값을 수정하였으면 다시 model 설정
            self.dt_max_depth.valueChanged.connect(self.decision_tree_model_update)

        elif model_name == 'Random Forest':
            # Random Forest 선택
            # hyperparameters : max_depth / n_estimators
            self.rf_max_depth = QSpinBox()
            self.rf_max_depth.setMinimum(1)
            self.rf_max_depth.setMaximum(1000)
            max_depth_layout = QHBoxLayout()
            max_depth_layout.addWidget(QLabel('Max Depth : '))
            max_depth_layout.addWidget(self.rf_max_depth)
            hyperparameter_layout.addLayout(max_depth_layout)

            # 값을 수정하였으면 다시 model 설정
            self.rf_max_depth.valueChanged.connect(self.random_forest_model_update)

            self.n_estimators = QSpinBox()
            self.n_estimators.setMinimum(1)
            self.n_estimators.setMaximum(1000)
            n_estimators_layout = QHBoxLayout()
            n_estimators_layout.addWidget(QLabel('n_estimators : '))
            n_estimators_layout.addWidget(self.n_estimators)
            hyperparameter_layout.addLayout(n_estimators_layout)

            self.n_estimators.valueChanged.connect(self.random_forest_model_update)

            # 값을 수정하였으면 다시 model 설정
            self.model = RandomForestClassifier(max_depth=self.rf_max_depth.value(), n_estimators=self.n_estimators.value())

    def logistic_model_update(self):
        self.model = LogisticRegression(max_iter=self.max_iters.value())

    def decision_tree_model_update(self):
        self.model = DecisionTreeClassifier(max_depth=self.dt_max_depth.value())

    def random_forest_model_update(self):
        self.model = RandomForestClassifier(max_depth=self.rf_max_depth.value(), n_estimators=self.n_estimators.value())

    def train(self):
        # 훈련하는 동안 팝업창(QDialog를 이용)을 띄우기 위해서 Threading을 사용
        self.popup = QDialog(self)
        self.popup.setWindowTitle("Training")
        layout = QVBoxLayout()
        self.label = QLabel("훈련이 끝나면 OK 버튼을 눌러주세요.")
        layout.addWidget(self.label)

        self.ok_button = QPushButton("OK")
        self.ok_button.setEnabled(False)    # 훈련하는 동안에는 ok버튼 비활성화
        self.ok_button.clicked.connect(self.on_ok_clicked)
        layout.addWidget(self.ok_button)

        self.popup.setLayout(layout)

        # 훈련을 Threading으로 병렬 처리
        self.training_thread = threading.Thread(target=self.run_training)
        self.training_thread.start()

        # 팝업 창 실행
        self.popup.exec_()

    def run_training(self):
        print("=======Modeling Start========")
        self.model.fit(self.X_train, self.y_train)  # 모델 훈련 시작

        # 예측값(label-encoding을 하였으면 다시 원래의 모습으로 변환)
        y_pred = self.model.predict(self.X_test)
        if self.label_encoder is not None:
            y_pred = self.label_encoder.inverse_transform(y_pred)
    
        # 결괏값 저장
        self.cr = classification_report(self.y_test, y_pred)
        self.cm = confusion_matrix(self.y_test, y_pred)
        self.accuracy = accuracy_score(self.y_test, y_pred)

        self.ok_button.setEnabled(True) # ok버튼 활성화

    def on_ok_clicked(self):
        # ok버튼을 눌렀을 때
        self.popup.accept()  # 팝업 창 닫기
        self.result()
        
    def result(self):
        print("=========RESULT==========")
        
        # 결괏값들 출력
        print(f"cr : {self.cr}")
        print(f"cm : {self.cm}")
        print(f"acc : {self.accuracy}")

        # acc score
        QMessageBox.about(self, "accuracy score", str(self.accuracy))

        # 혼동행렬 plot 함수
        def plot_confusion_matrix(cm, classes):
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=classes, yticklabels=classes)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Confusion Matrix')
            plt.show()

        class_names = list(self.y_test.unique())    # y_test의 클래스 이름들 class_names에 저장
        plot_confusion_matrix(self.cm, class_names) # 혼동행렬 plot

    # layout clear해주는 함수
    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)

            if item.widget():
                item.widget().deleteLater()

            elif item.layout():
                self.clear_layout(item.layout())

    # qt창 화면 가운데 정렬해주는 함수
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ml_app = ML_app()
    ml_app.show()
    sys.exit(app.exec_())