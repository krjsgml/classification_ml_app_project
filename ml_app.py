import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from io import StringIO
import threading

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import Qt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from keras._tf_keras.keras import Input
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.utils import plot_model

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


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

        self.target_variable_selection = None
        self.graph_variable_selection = None
        self.drop_variable_selection = None
        self.select_col_value_counts = None

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

        # 수치형 자료에서 엔코딩 되어있는 변수 추출
        self.exclude_encoding_cols = []
        self.saved_checkbox_states = []
        self.suspicious_checkboxes = []
        self.dialog = None

        suspicious_encoding_layout = QHBoxLayout()
        suspicious_encoding_layout.addWidget(QLabel("Select the variables suspected of encoding"))
        self.suspicious_encoding_btn = QPushButton("Check")
        self.suspicious_encoding_btn.setEnabled(False)
        suspicious_encoding_layout.addWidget(self.suspicious_encoding_btn)
        self.suspicious_encoding_btn.clicked.connect(self.open_suspicious_encoding_dialog)

        EDA_layout.addLayout(EDA_menu_layout, 1)
        EDA_layout.addLayout(select_target_variable_layout, 1)
        EDA_layout.addLayout(select_show_graph_variable_layout, 1)
        EDA_layout.addLayout(select_drop_variable_layout, 1)
        EDA_layout.addLayout(suspicious_encoding_layout, 1)
        EDA_layout.addStretch(1)
        EDA_layout.addLayout(EDA_show_layout, 4)

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
        info.clicked.connect(lambda : self.display_info(EDA_show_layout, data_info))

        # describe 버튼 클릭 시 describe() 메서드를 이용해 앱에 출력
        describe.clicked.connect(lambda : self.clear_layout(EDA_show_layout))
        describe.clicked.connect(lambda : self.display_describe(EDA_show_layout, data_describe))

        # value_counts 버튼 클릭 시 value_counts() 메서드를 이용해 앱에 출력
        value_counts.clicked.connect(lambda : self.clear_layout(EDA_show_layout))
        value_counts.clicked.connect(lambda : self.select_value_counts(EDA_show_layout, self.step1_data))

    def display_info(self, layout, info):
        # info 버튼을 클릭 시 고정폭 폰트를 이용해 가독성 향상
        text_edit = QPlainTextEdit()
        text_edit.setPlainText(info)
        text_edit.setReadOnly(True)
        
        font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        text_edit.setFont(font)
        
        layout.addWidget(text_edit)

    def display_describe(self, layout, df):
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

    def display_value_counts(self, layout, df, col):
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

    def select_value_counts(self, layout, df):
        # 각각의 변수들 하나하나 value_counts()를 확인하고자 QCombox를 만들어 변수를 선택하면
        # display_value_counts()가 호출되어 선택한 변수의 value_counts()를 확인할 수 있음.
        select_col_layout = QVBoxLayout()
        show_value_counts_layout = QHBoxLayout()

        self.select_col_value_counts = QComboBox()
        select_col_layout.addWidget(self.select_col_value_counts)

        layout.addLayout(select_col_layout)
        layout.addLayout(show_value_counts_layout)

        self.select_col_value_counts.addItems(df.columns)
        self.select_col_value_counts.currentTextChanged.connect(lambda col : self.display_value_counts(show_value_counts_layout, df, col))

    def show_graph(self):
        # Qcombobox에서 선택한 변수의 그래프를 출력
        variable = self.graph_variable_selection.currentText()
        if variable != "Select Variable":
            print(f"Show Graph Variable : {variable}")
            if self.step1_data[variable].dtype in ['int', 'float']: # 수치형은 histplot 그래프
                sns.histplot(data=self.step1_data, x=self.step1_data[variable])

            else:   # 범주형은 bar 그래프
                self.step1_data[variable].value_counts().plot(kind='bar', alpha=0.7)

            plt.savefig("project_1/ml_app/data_preprocessing/variable_graph.png")
            plt.close()

        variable_graph_dialog = QDialog(self)
        variable_graph_dialog.setWindowTitle("Variable Graph")
        variable_graph_dialog.resize(800, 600)
        variable_layout = QVBoxLayout()
        variable_graph_dialog.setLayout(variable_layout)

        pixmap = QPixmap("project_1/ml_app/data_preprocessing/variable_graph.png")
        label = QLabel(variable_graph_dialog)
        pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        variable_layout.addWidget(label)

        variable_graph_dialog.exec_()


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

            if self.select_col_value_counts is not None:
                value_counts_col_index = self.select_col_value_counts.findText(variable)
                self.select_col_value_counts.removeItem(value_counts_col_index)
            
            QMessageBox.about(self, "Delete Variable", f"drop {variable}")

    def target_variable_split(self, text):
        # 타겟 변수 선택 시 self.target_variable로 저장
        # x,y 분리는 이상치와 결측치 제거 후 진행
        if text != 'Select Variable':
            print(f"Target Variable : {text}")
            self.target_variable = text

            # 엔코딩된 변수를 선택하게 하기
            self.suspicious_encoding_btn.setEnabled(True)
            
    def open_suspicious_encoding_dialog(self):
        self.dialog = QDialog(self)
        self.dialog.setWindowTitle("원-핫/라벨 엔코딩 의심 변수 선택")
        self.dialog.accepted.connect(lambda: self.suspicious_checkboxes.clear())
        self.dialog.rejected.connect(lambda: self.suspicious_checkboxes.clear())

        layout = QVBoxLayout()
        self.dialog.setLayout(layout)

        layout.addWidget(QLabel("의심되는 수치형 변수 선택: "))

        scroll_area = QScrollArea(self.dialog)
        scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # feature변수의 수치형 자료 리스트
        features = self.step1_data.drop(columns=[self.target_variable])
        num_features = features.select_dtypes(include=['int', 'float']).columns.tolist()
        
        for col in num_features:
            # 원핫, 라벨 엔코딩이 의심되는 변수(column) suspicious_checkboxes에 추가하여 체크박스에 추가
            if self.is_onehot(col) or self.is_label(col):
                checkbox = QCheckBox(col)
                self.suspicious_checkboxes.append(checkbox)

                # 체크박스 상태 복원
                if col in self.get_suspicious_checkbox_states():  # 메서드를 호출
                    checkbox.setChecked(True)

                scroll_layout.addWidget(checkbox)
                
        if len(self.suspicious_checkboxes) == 0:
            self.step2.setEnabled(True)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # 확인 버튼
        check_ok_btn = QPushButton("OK")
        check_ok_btn.clicked.connect(self.apply_selected_suspicious_vars)
        layout.addWidget(check_ok_btn)

        self.dialog.exec_()

    def apply_selected_suspicious_vars(self):
        self.exclude_encoding_cols.clear()  # 중복 없애기 위해서 clear

        for checkbox in self.suspicious_checkboxes:
            if checkbox.isChecked():
                self.exclude_encoding_cols.append(checkbox.text())

                # 체크박스상태저장 리스트에 없다면 추가
                if checkbox.text() not in self.saved_checkbox_states:
                    self.saved_checkbox_states.append(checkbox.text())
            
            # 체크박스상태저장 리스트에 있지만 체크가 해제된 경우에 삭제
            else:
                if checkbox.text() in self.saved_checkbox_states:
                    self.saved_checkbox_states.remove(checkbox.text())

        self.dialog.accept()
        # 다음 step 이동 가능
        self.step2.setEnabled(True)

    def is_onehot(self, col):
        nunique_vals = self.step1_data[col].nunique()
        unique_vals = self.step1_data[col].unique()
        return (nunique_vals == 2) and sorted(list(unique_vals)) == [0, 1]
    
    def is_label(self, col):
        unique_vals = sorted(float(val) for val in self.step1_data[col].unique())
        unique_vals_1 = sorted(float(val-1) for val in self.step1_data[col].unique())
        return (unique_vals == list(map(float, range(len(unique_vals))))) or ((unique_vals_1 == list(map(float, range(len(unique_vals)))))) 
    
    def get_suspicious_checkbox_states(self):
        # 현재 체크박스 상태를 반환하는 메서드
        return self.saved_checkbox_states

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
        self.missing_outlier_ok.clicked.connect(self.missing_outlier_preprocess)

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

        print(f"Find col have outlier value {str(outlier_cols)}")

        # 이상치 발견
        if len(outlier_cols) > 0:
            self.outlier_found = 1

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

        # X, y 분리 및 train_test_split 
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
        
        # exclude cols (already encoding cols)
        for cols in self.exclude_encoding_cols:
            self.nums_X_train.remove(cols)
            self.nums_X_test.remove(cols)

        print(self.nums_X_train)

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

        # 엔코딩이 안되는 col은 데이터셋에서 삭제
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
                encoder = OneHotEncoder(sparse_output=False)
                try:
                    for col in self.objs_X_train:
                        transformed_train = encoder.fit_transform(self.X_train[[col]])
                        encoded_cols = [f"{col}_{category}" for category in encoder.categories_[0]]
                        transformed_train_df = pd.DataFrame(transformed_train, columns=encoded_cols, index=self.X_train.index)

                        transformed_test = encoder.transform(self.X_test[[col]])
                        transformed_test_df = pd.DataFrame(transformed_test, columns=encoded_cols, index=self.X_test.index)

                        self.X_train = pd.concat([self.X_train, transformed_train_df], axis=1).drop(columns=[col])
                        self.X_test = pd.concat([self.X_test, transformed_test_df], axis=1).drop(columns=[col])

                except:
                        self.X_train = self.X_train.drop(columns=[col])
                        self.X_test = self.X_test.drop(columns=[col])
                        QMessageBox.about(self, "error", f"drop {col}")

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
        plt.savefig("project_1/ml_app/data_preprocessing/data_imbalance.png")
        plt.close()

        plot_target_graph_dialog = QDialog(self)
        plot_target_graph_dialog.setWindowTitle("Data Imbalance")
        plot_target_graph_dialog.resize(800, 600)
        target_graph_layout = QVBoxLayout()
        plot_target_graph_dialog.setLayout(target_graph_layout)

        pixmap = QPixmap("project_1/ml_app/data_preprocessing/data_imbalance.png")
        label = QLabel(plot_target_graph_dialog)
        pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        target_graph_layout.addWidget(label)

        plot_target_graph_dialog.exec_()

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
        print("\n=========== Modeling ==========\n")
        # model 선택
        # Logistic Regression / Decision Tree / Random Forest
        self.model_selection = QComboBox(self)
        self.model_selection.addItem('model selection')
        self.model_selection.addItem('Logistic Regression')
        self.model_selection.addItem('Decision Tree')
        self.model_selection.addItem('Random Forest')
        self.model_selection.addItem("DNN")

        self.model_selection_layout = QHBoxLayout()
        self.model_selection_layout.addWidget(QLabel("Model Selection : "))
        self.model_selection_layout.addWidget(self.model_selection)
        self.model_setup_layout.addLayout(self.model_selection_layout)

        # 모델이 선택된다면 hyperparamer를 설정할 수 있게 해줌
        self.model_selection.activated[str].connect(self.hyperparameter_setup)

    def hyperparameter_setup(self, model_name):
        # 모델의 하이퍼파라미터를 설정
        self.clear_layout(self.model_hyperparameter_layout)
        self.hyperparameter_layout = QVBoxLayout()
        train_layout = QVBoxLayout()
        self.model_hyperparameter_layout.addLayout(self.hyperparameter_layout, 7)
        self.model_hyperparameter_layout.addLayout(train_layout, 3)

        train_model = QPushButton("Train Model")
        train_layout.addWidget(train_model)
        train_model.clicked.connect(self.train)

        if self.model_selection_layout.count() == 3:
            dnn_set_btn = self.model_selection_layout.itemAt(self.model_selection_layout.count() - 1)
            dnn_set_widget = dnn_set_btn.widget()
            self.model_selection_layout.removeWidget(dnn_set_widget)
            dnn_set_widget.deleteLater()
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
            self.hyperparameter_layout.addLayout(max_iters_layout)

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
            self.hyperparameter_layout.addLayout(max_depth_layout)

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
            self.hyperparameter_layout.addLayout(max_depth_layout)

            # 값을 수정하였으면 다시 model 설정
            self.rf_max_depth.valueChanged.connect(self.random_forest_model_update)

            self.n_estimators = QSpinBox()
            self.n_estimators.setMinimum(1)
            self.n_estimators.setMaximum(1000)
            n_estimators_layout = QHBoxLayout()
            n_estimators_layout.addWidget(QLabel('n_estimators : '))
            n_estimators_layout.addWidget(self.n_estimators)
            self.hyperparameter_layout.addLayout(n_estimators_layout)

            self.n_estimators.valueChanged.connect(self.random_forest_model_update)

            # 값을 수정하였으면 다시 model 설정
            self.model = RandomForestClassifier(max_depth=self.rf_max_depth.value(), n_estimators=self.n_estimators.value())

        elif model_name == "DNN":
            self.dnn_hyperparams_layout = None
            setting_dnn_model = QPushButton("Setting model")
            setting_dnn_model.clicked.connect(self.DNN_modeling)
            
            self.model_selection_layout.addWidget(setting_dnn_model)

    def DNN_modeling(self):
        self.dnn_dialog = QDialog(self)
        self.dnn_dialog.setWindowTitle("Setting DNN Model")
        self.dnn_layout = QVBoxLayout()
        self.dnn_dialog.setLayout(self.dnn_layout)

        self.layer_cnt = 1

        self.add_layer()

        self.add_layer_btn = QPushButton("Add Dense Layer")
        self.add_layer_btn.clicked.connect(self.add_dense_layer)
        self.dnn_layout.addWidget(self.add_layer_btn)

        # DNN 모델 설정이 끝난 후 Plot Model 버튼 추가
        self.plot_model_button = QPushButton("Plot Model")
        self.plot_model_button.clicked.connect(self.plot_dnn_model)
        self.dnn_layout.addWidget(self.plot_model_button)

        # OK 버튼 추가
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.close_dnn_dialog)
        ok_button.clicked.connect(self.dnn_hyperparameter_setting)
        
        self.dnn_layout.addWidget(ok_button)

        self.dnn_dialog.setLayout(self.dnn_layout)

        self.dnn_dialog.exec_()

    def close_dnn_dialog(self):
        self.dnn_dialog.accept()

    def dnn_hyperparameter_setting(self):
        if self.dnn_hyperparams_layout is not None:
            self.clear_layout(self.dnn_hyperparams_layout)

        self.dnn_hyperparams_layout = QVBoxLayout()
        dnn_epochs_layout = QHBoxLayout()
        dnn_batchsize_layout = QHBoxLayout()
        dnn_valdata_layout = QHBoxLayout()
        dnn_callbacks = QHBoxLayout()

        self.dnn_hyperparams_layout.addLayout(dnn_epochs_layout)
        self.dnn_hyperparams_layout.addLayout(dnn_batchsize_layout)
        self.dnn_hyperparams_layout.addLayout(dnn_valdata_layout)
        self.dnn_hyperparams_layout.addLayout(dnn_callbacks)

        self.hyperparameter_layout.addLayout(self.dnn_hyperparams_layout)

        self.epochs_set = QSpinBox()
        self.epochs_set.setMinimum(1)
        self.epochs_set.setMaximum(1000)
        dnn_epochs_layout.addWidget(QLabel("epochs : "))
        dnn_epochs_layout.addWidget(self.epochs_set)
        self.epochs = self.epochs_set.value()
        self.epochs_set.valueChanged.connect(self.dnn_hyperparameters_setting)
        
        self.batch_size_set = QSpinBox()
        self.batch_size_set.setMinimum(1)
        self.batch_size_set.setMaximum(1000)
        dnn_batchsize_layout.addWidget(QLabel("batch size : "))
        dnn_batchsize_layout.addWidget(self.batch_size_set)
        self.batch_size = self.batch_size_set.value()
        self.batch_size_set.valueChanged.connect(self.dnn_hyperparameters_setting)

        self.validation_size_set = QSpinBox()
        self.validation_size_set.setMinimum(1)
        self.validation_size_set.setMaximum(100)
        dnn_valdata_layout.addWidget(QLabel("validation data ratio : "))
        dnn_valdata_layout.addWidget(self.validation_size_set)
        self.validation_size = float(self.validation_size_set.value()/100)
        self.validation_size_set.valueChanged.connect(self.dnn_hyperparameters_setting)

        self.earlystopping = QCheckBox("early stopping")
        self.checkpoint = QCheckBox("check point")
        self.earlystopping.stateChanged.connect(self.dnn_hyperparameters_setting)
        self.checkpoint.stateChanged.connect(self.dnn_hyperparameters_setting)
        dnn_callbacks.addWidget(self.earlystopping)
        dnn_callbacks.addWidget(self.checkpoint)

    def dnn_hyperparameters_setting(self):
        self.epochs = self.epochs_set.value()
        self.batch_size = self.batch_size_set.value()
        self.validation_size = float(self.validation_size_set.value() / 100)

        self.callbacks = []
        if self.earlystopping.isChecked():
            early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)
            self.callbacks.append(early_stopping_callback)
        if self.checkpoint.isChecked():
            checkpoint_callback = ModelCheckpoint('project_1/ml_app/dnn_model/best_model.keras', verbose=1,
                                         monitor="val_loss", mode="min", save_best_only=True)
            self.callbacks.append(checkpoint_callback)

    def add_layer(self):
        # 레이어에 대한 레이아웃
        layer_layout = QHBoxLayout()

        # 레이어 이름 라벨
        layer_label = QLabel(f'Dense{self.layer_cnt}')
        layer_layout.addWidget(layer_label)

        # 뉴런 갯수 선택 스핀박스
        neuron_spinbox = QSpinBox()
        set_neuron_num = len(np.unique(self.y_train))
        if set_neuron_num == 2:
            set_neuron_num -= 1
        neuron_spinbox.setRange(1, 1024)  # 뉴런 갯수 설정 범위
        neuron_spinbox.setValue(set_neuron_num)
        layer_layout.addWidget(neuron_spinbox)

        # 활성화 함수 선택 콤보박스
        activation_combo = QComboBox()
        activation_combo.addItems(['relu', 'sigmoid', 'tanh', 'softmax'])
        layer_layout.addWidget(activation_combo)

        # Dropout 체크박스
        dropout_checkbox = QCheckBox('Dropout')
        dropout_checkbox.stateChanged.connect(lambda state, box=dropout_checkbox: self.add_dropout_option(state, layer_layout))
        layer_layout.addWidget(dropout_checkbox)

         # 'X' 버튼 추가
        if self.layer_cnt != 1:
            remove_button = QPushButton('X')
            remove_button.clicked.connect(lambda: self.remove_layer(layer_layout))
            layer_layout.addWidget(remove_button)

        # 레이아웃에 레이어 추가
        self.dnn_layout.insertLayout(self.dnn_layout.count()-3, layer_layout)

    def remove_layer(self, layer_layout):
        self.layer_cnt -= 1
        # 레이아웃을 삭제
        for i in reversed(range(layer_layout.count())):
            widget = layer_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.dnn_layout.removeItem(layer_layout)
    
    def add_dense_layer(self):
        self.layer_cnt += 1
        self.add_layer()

    def add_dropout_option(self, state, layer_layout):
        # Dropout 비율 설정 스핀박스를 레이아웃에 추가하고, 따로 저장해둠
        if state == 2:  # 체크박스가 선택되면
            # Dropout 비율 설정 스핀박스 추가
            dropout_spinbox = QDoubleSpinBox()
            dropout_spinbox.setRange(0.0, 0.9)
            dropout_spinbox.setSingleStep(0.1)
            dropout_spinbox.setValue(0.5)
            dropout_spinbox.setObjectName("dropout_spinbox")
            layer_layout.addWidget(dropout_spinbox)
        else:
            # 체크 해제되면 Dropout 비율 제거
            for i in range(layer_layout.count()):
                widget = layer_layout.itemAt(i).widget()
                if isinstance(widget, QDoubleSpinBox):
                    widget.deleteLater()
    
    def plot_dnn_model(self):
        plot_dialog = QDialog(self)
        plot_dialog.setWindowTitle("DNN Model Plot")
        plot_layout = QVBoxLayout()
        plot_dialog.setLayout(plot_layout)
    
        self.model = Sequential()
        self.model.add(Input(shape=(self.X_train.shape[1],)))
    
        for i in range(self.dnn_layout.count() - 3):  # 'Add Dense Layer' 버튼과 'OK' 버튼 제외
            layer_info = self.dnn_layout.itemAt(i).layout()
            neurons = layer_info.itemAt(1).widget().value()  # 뉴런 수
            activation = layer_info.itemAt(2).widget().currentText()  # 활성화 함수
            self.model.add(Dense(neurons, activation=activation))
    
            # Dropout 체크박스 확인
            dropout_checkbox = layer_info.itemAt(3).widget()
            if dropout_checkbox.isChecked():  # Dropout이 체크된 경우
                # ObjectName으로 스핀박스를 찾아서 Dropout 비율 추가
                for j in range(layer_info.count()):
                    widget = layer_info.itemAt(j).widget()
                    if isinstance(widget, QDoubleSpinBox):  # Dropout 비율 설정
                        dropout_rate = widget.value()
                        self.model.add(Dropout(dropout_rate))  # Dropout 레이어 추가
                        break
                    
        self.model.summary()
    
        img_label = QLabel(plot_dialog)
    
        plot_model(self.model, to_file='project_1/ml_app/dnn_model/dnn_model.png', show_shapes=True)
        pixmap = QPixmap("project_1/ml_app/dnn_model/dnn_model.png")
        pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    
        img_label.setPixmap(pixmap)
        plot_layout.addWidget(img_label)
    
        plot_dialog.exec_()

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

        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=self.validation_size, stratify=self.y_train)
        classes = len(np.unique(y_train))
        if isinstance(self.model, Sequential):
            # 다중 분류 문제
            if classes != 2:
                print("sparse")
                self.model.compile(optimizer="adam",
                                   loss="sparse_categorical_crossentropy",
                                   metrics=['accuracy'])
            # 이진 분류 문제
            else:
                print("binary")
                self.model.compile(optimizer="adam",
                                   loss="binary_crossentropy",
                                   metrics=['accuracy'])
                
            self.history = self.model.fit(x=X_train, y=y_train, epochs=self.epochs, batch_size=self.batch_size,
                                     validation_data=(X_val, y_val), verbose=1,
                                     callbacks=self.callbacks)
            
            y_pred = self.model.predict(self.X_test)
            y_pred = np.argmax(y_pred, axis=1)

        else:   
            self.model.fit(self.X_train, self.y_train)  # 모델 훈련 시작
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
            plt.savefig("project_1/ml_app/model_result/confusion_matrix.png")
            plt.close()
            
            cm_dialog = QDialog(self)
            cm_dialog.setWindowTitle("Confusion Matrix")
            cm_dialog.resize(800, 600)
            cm_layout = QVBoxLayout()
            cm_dialog.setLayout(cm_layout)

            pixmap = QPixmap("project_1/ml_app/model_result/confusion_matrix.png")
            label = QLabel(cm_dialog) 
            pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pixmap)
            cm_layout.addWidget(label)

            cm_dialog.exec_()

        # model이 decision tree이면 plot_tree()로 tree를 보여줌
        if isinstance(self.model, DecisionTreeClassifier):
            plt.figure(figsize=(20, 20))
            plot_tree(self.model, filled=True, feature_names=self.X_train.columns, class_names=class_names, rounded=True)
            plt.title("Deicision Tree")
            plt.savefig("project_1/ml_app/tree_plot/decision_tree.png")
            plt.close()

            dt_dialog = QDialog(self)
            dt_dialog.setWindowTitle("Decision Tree")
            dt_dialog.resize(800 ,600)
            dt_layout = QVBoxLayout()
            dt_dialog.setLayout(dt_layout)

            pixmap = QPixmap("project_1/ml_app/tree_plot/decision_tree.png")
            label = QLabel(dt_dialog)
            pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pixmap)
            dt_layout.addWidget(label)

            dt_dialog.exec_()
        
        if isinstance(self.model, Sequential):
            # DNN Graph Dialog
            dnn_graph_dialog = QDialog(self)
            dnn_graph_dialog.setWindowTitle("DNN Graph")
            dnn_graph_dialog.resize(800, 600)  # Dialog 크기 설정
            layout = QVBoxLayout()
            dnn_graph_dialog.setLayout(layout)
    
            # 그래프 그리기
            plt.plot(self.history.history['accuracy'], label='Train')
            plt.plot(self.history.history['val_accuracy'], label='Validation')  # 쌍따옴표 수정
            plt.title("DNN Model Accuracy")  # 제목 수정
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(loc='best')
            plt.savefig("project_1/ml_app/dnn_model/dnn_accuracy.png")
            plt.close()
    
            # Pixmap 설정
            pixmap = QPixmap("project_1/ml_app/dnn_model/dnn_accuracy.png")
            label = QLabel(dnn_graph_dialog)  # QLabel의 부모를 dnn_graph_dialog로 수정
            pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pixmap)
            layout.addWidget(label)
    
            dnn_graph_dialog.exec_()

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