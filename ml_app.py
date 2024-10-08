import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt
from io import StringIO
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns


class ML_app(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selected_data = None
        self.target_variable = None
        self.model = None

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

        # 레이아웃 비율 설정 (예: leftlayout 70%, rightlayout 30%)
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
        menubar = self.menuBar()
        data_menu = menubar.addMenu('&Data')
        self.load_data_menu(data_menu)

    def load_data_menu(self, menu):
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
            self.selected_data = pd.read_csv(data_path)
            QMessageBox.about(self, 'Load Data', f'Data loaded: {filename}')
            self.image_label.hide()     # Hide main image
            self.preprocess()
        
    def preprocess(self):
        preprocessing_layout = QVBoxLayout()
        step1 = QPushButton('STEP1')
        step2 = QPushButton('STEP2')
        step3 = QPushButton('STEP3')
        step4 = QPushButton('STEP4')

        preprocessing_layout.addWidget(step1)
        preprocessing_layout.addWidget(step2)
        preprocessing_layout.addWidget(step3)
        preprocessing_layout.addWidget(step4)

        step1.clicked.connect(self.EDA)
        step2.clicked.connect(self.missing_outlier_value)
        step3.clicked.connect(self.data_scale_encoding)
        step4.clicked.connect(self.class_imbalance)

        self.preprocess_step_layout.addLayout(preprocessing_layout)

    def EDA(self):
        self.clear_layout(self.preprocess_layout)

        # 고유한 칼럼 삭제 list에 추가
        self.drop_cols = []
        for col in self.selected_data.columns:
            if self.selected_data[col].nunique() == len(self.selected_data):
                self.drop_cols.append(col)
        #self.selected_data.drop(columns=drop_cols, inplace=True)

        EDA_layout = QVBoxLayout()
        EDA_menu_layout = QHBoxLayout()
        EDA_show_layout = QVBoxLayout()
        
        # 타겟변수 설정
        select_target_variable_layout = QHBoxLayout()
        select_target_variable_layout.addWidget(QLabel("Select Target Variable :"))

        self.target_variable_selection = QComboBox(self)
        select_target_variable_layout.addWidget(self.target_variable_selection)

        self.target_variable_selection.addItem('Select Variable')
        self.target_variable_selection.addItems(self.selected_data.columns)
        self.target_variable_selection.activated[str].connect(self.target_variable_split)


        # 변수 그래프 보여주기
        select_show_graph_variable_layout = QHBoxLayout()
        select_show_graph_variable_layout.addWidget(QLabel("Select Show Graph :"))

        self.graph_variable_selection = QComboBox(self)
        select_show_graph_variable_layout.addWidget(self.graph_variable_selection)

        self.graph_variable_selection.addItem("Select Variable")
        self.graph_variable_selection.addItems(self.selected_data.columns)

        self.show_graph_btn = QPushButton("Show")
        select_show_graph_variable_layout.addWidget(self.show_graph_btn)
        self.show_graph_btn.clicked.connect(self.show_graph)

        # 변수 삭제
        select_drop_variable_layout = QHBoxLayout()
        select_drop_variable_layout.addWidget(QLabel("Select Drop Variable :"))
        
        self.drop_variable_selection = QComboBox(self)
        select_drop_variable_layout.addWidget(self.drop_variable_selection)

        self.drop_variable_selection.addItem("Select Variable")
        self.drop_variable_selection.addItems(self.selected_data.columns)
        
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

        buffer = StringIO()
        self.selected_data.info(buf=buffer)
        data_info = buffer.getvalue()

        data_describe = str(self.selected_data.describe())
        data_value_counts = str(self.selected_data.value_counts())

        info.clicked.connect(lambda : self.clear_layout(EDA_show_layout))
        info.clicked.connect(lambda : EDA_show_layout.addWidget(QLabel(data_info)))
        describe.clicked.connect(lambda : self.clear_layout(EDA_show_layout))
        describe.clicked.connect(lambda : EDA_show_layout.addWidget(QLabel(data_describe)))
        value_counts.clicked.connect(lambda : self.clear_layout(EDA_show_layout))
        value_counts.clicked.connect(lambda : EDA_show_layout.addWidget(QLabel(data_value_counts)))

    def show_graph(self):
        variable = self.graph_variable_selection.currentText()
        if variable != "Select Variable":
            if self.selected_data[variable].dtype in ['int', 'float']:
                sns.histplot(data=self.selected_data, x=self.selected_data[variable])

            else:
                self.selected_data[variable].value_counts().plot(kind='bar', alpha=0.7)

            plt.show()

    def drop_ok(self):
        variable = self.drop_variable_selection.currentText()
        if variable != "Select Variable":
            self.selected_data.drop(columns=[variable], inplace=True)
            target_col_index = self.target_variable_selection.findText(variable)
            graph_col_index = self.graph_variable_selection.findText(variable)
            drop_col_index = self.drop_variable_selection.findText(variable)
            self.target_variable_selection.removeItem(target_col_index)
            self.graph_variable_selection.removeItem(graph_col_index)
            self.drop_variable_selection.removeItem(drop_col_index)

            QMessageBox.about(self, "Delete Variable", f"drop {variable}")

    def target_variable_split(self, text):
        # x,y 분리는 이상치와 결측치 제거 후 진행
        if text != 'Select Variable':
            self.target_variable = text

    def missing_outlier_value(self):
        if self.target_variable_selection.currentText() == "Select Variable":
            QMessageBox.about(self, "Error", "Please select target variable")

        else:
            self.clear_layout(self.preprocess_layout)
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

            # missing value
            missing_value_cnt = self.selected_data.isnull().sum().sum()
            if missing_value_cnt > 0:
                self.missing_found = 1

            if self.missing_found:
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

            for col in self.selected_data.select_dtypes(include=['float', 'int']):
                lower, upper = self.outlier_bound(col)
                if ((self.selected_data[col] < lower) | (self.selected_data[col] > upper)).any():
                    self.outlier_found = 1

            if self.outlier_found:
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
        error = 0
        if self.missing_found:
            if self.missing_val_delete.isChecked():
                self.selected_data.dropna(inplace=True)

            elif self.missing_val_replace.isChecked():
                nums = self.selected_data.select_dtypes(include=['int', 'float']).columns.tolist()
                objs = self.selected_data.select_dtypes(include=['object']).columns.tolist()

                for num in nums:
                    self.selected_data[num] = self.selected_data[num].fillna(self.selected_data[num].mean())
                for obj in objs:
                    self.selected_data[obj] = self.selected_data[obj].fillna(self.selected_data[obj].mode())

            else:
                error = 1

        if self.outlier_found:
            if self.outlier_val_delete.isChecked():
                for col in self.selected_data.select_dtypes(include=['float', 'int']).columns:
                    lower, upper = self.outlier_bound(col)
                    self.selected_data = self.selected_data[(self.selected_data[col] >= lower) & (self.selected_data[col] <= upper)]

            elif self.outlier_val_replace.isChecked():
                 for col in self.selected_data.select_dtypes(include=['float', 'int']).columns:
                    lower, upper = self.outlier_bound(col)
                    col_min = self.selected_data[col].min()
                    col_max = self.selected_data[col].max()

                    self.selected_data[col] = self.selected_data[col].apply(
                        lambda x: col_min if x < lower else (col_max if x > upper else x)
                    )
            else:
                error = 1
        
        if error == 1:
            QMessageBox.about(self, "erorr", "select please")

        else:
            self.clear_layout(self.preprocess_layout)
            self.preprocess_layout.addWidget(QLabel("Go Next Step"))
            QMessageBox.about(self, "complete", "missing val & outlier val preprocess complete")

    def outlier_bound(self, col):
        Q1 = self.selected_data[col].quantile(0.25)
        Q3 = self.selected_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return lower, upper
    
    def data_scale_encoding(self):
        self.clear_layout(self.preprocess_layout)
        self.preprocess_layout.addWidget(QLabel("train-test-split ratio"))
        self.train_test_split_input = QSpinBox()
        self.preprocess_layout.addWidget(self.train_test_split_input)
        self.train_test_split_input.setMaximum(100)
        self.train_test_split_input.setMinimum(0)
        self.train_test_split_input.setValue(70)

        self.X = self.selected_data.drop(columns=[self.target_variable])
        self.y = self.selected_data[self.target_variable]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=float(self.train_test_split_input.value()/100), random_state=42, stratify=self.y
        )
        self.train_test_split_input.valueChanged.connect(self.train_test_split_)

        data_scale_layout = QHBoxLayout()
        data_encoding_layout = QHBoxLayout()
        self.preprocess_layout.addLayout(data_scale_layout)
        self.preprocess_layout.addLayout(data_encoding_layout)
        self.nums_X_train = list(self.X_train.select_dtypes(include=['float', 'int']).columns)
        self.nums_X_test = list(self.X_test.select_dtypes(include=['float', 'int']).columns)

        self.objs_X_train = list(self.X_train.select_dtypes(include=['object']).columns)
        self.objs_X_test = list(self.X_test.select_dtypes(include=['object']).columns)

        self.scale_encoding_ok = QPushButton("OK")
        self.preprocess_layout.addStretch(2)
        self.preprocess_layout.addWidget(self.scale_encoding_ok)
        self.scale_encoding_ok.clicked.connect(self.scale_encoding_preprocess)

        if len(self.nums_X_train) > 0:
            select_scale = QVBoxLayout()
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
            select_encoding = QVBoxLayout()
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
        error = 0
    
        if len(self.nums_X_train) > 0:
            if self.normalization.isChecked():
                scaler = MinMaxScaler()
                self.X_train[self.nums_X_train] = scaler.fit_transform(self.X_train[self.nums_X_train])
                self.X_test[self.nums_X_test] = scaler.transform(self.X_test[self.nums_X_test])
            
            elif self.standardization.isChecked():
                scaler = StandardScaler()
                self.X_train[self.nums_X_train] = scaler.fit_transform(self.X_train[self.nums_X_train])
                self.X_test[self.nums_X_test] = scaler.transform(self.X_test[self.nums_X_test])

            else:
                error = 1

        if len(self.objs_X_train) > 0:
            if self.label_encoding.isChecked():
                encoder = LabelEncoder()
                for col in self.objs_X_train:
                    try:
                        self.X_train[col] = encoder.fit_transform(self.X_train[[col]])
                        self.X_test[col] = encoder.transform(self.X_test[col])
                    except:
                        self.X_train = self.X_train.drop(columns=[col])
                        self.X_test = self.X_test.drop(columns=[col])
                        QMessageBox.about(self, "error", f"drop {col}")

            elif self.one_hot_encoding.isChecked():
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

                for col in self.objs_X_train:
                    transformed_train = encoder.fit_transform(self.X_train[[col]])
                    encoded_cols = [f"{col}_{category}" for category in encoder.categories_[0]]
                    transformed_train_df = pd.DataFrame(transformed_train, columns=encoded_cols, index=self.X_train.index)

                    transformed_test = encoder.transform(self.X_test[[col]])
                    transformed_test_df = pd.DataFrame(transformed_test, columns=encoded_cols, index=self.X_test.index)

                    self.X_train = pd.concat([self.X_train, transformed_train_df], axis=1).drop(columns=[col])
                    self.X_test = pd.concat([self.X_test, transformed_test_df], axis=1).drop(columns=[col])

            else:
                error = 1

        # 이미 엔코딩이 되어있는 경우를 위해 y_train이 object type이면 엔코딩하기
        if self.y_train.dtype == 'object':
            self.label_encoder = LabelEncoder()
            self.y_train = self.label_encoder.fit_transform(self.y_train)
            self.y_train = pd.Series(self.y_train)

        if error == 1:
            QMessageBox.about(self, "error", "select please")

        else:
            self.clear_layout(self.preprocess_layout)
            self.preprocess_layout.addWidget(QLabel("Go Next Step"))
            QMessageBox.about(self, "complete", "scaling & encoding complete")

    def train_test_split_(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=float(self.train_test_split_input.value()/100), random_state=42, stratify=self.y
        )

    def class_imbalance(self):
        self.clear_layout(self.preprocess_layout)

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
        class_imbalance_method_layout.addWidget(oversampling)
        class_imbalance_method_layout.addWidget(undersampling)
        sampling_ok = QPushButton("OK")
        class_imbalance_method_layout.addWidget(sampling_ok)

        self.plot_target_variable(self.y_train)
        class_imbalance_layout.addStretch(2)

        oversampling.clicked.connect(self.oversampling_process)
        undersampling.clicked.connect(self.undersampling_precess)
        sampling_ok.clicked.connect(self.sampling_process_ok)

    def plot_target_variable(self, target):
        class_counts = target.value_counts()

        class_counts.plot(kind='bar')
        plt.title('train - target variable')
        plt.xlabel('Class')
        plt.ylabel('Freq')
        plt.show()

    def oversampling_process(self):
        self.sampling_x = self.X_train.copy()
        self.sampling_y = self.y_train.copy()
        self.sampling_x, self.sampling_y = RandomOverSampler(random_state=0).fit_resample(self.sampling_x, self.sampling_y)
        self.sampling_y = pd.DataFrame(self.sampling_y, columns=[self.target_variable])
       
        self.plot_target_variable(self.sampling_y)

    def undersampling_precess(self):
        self.sampling_x = self.X_train.copy()
        self.sampling_y = self.y_train.copy()
        self.sampling_x, self.sampling_y = RandomUnderSampler(random_state=0).fit_resample(self.sampling_x, self.sampling_y)
        self.sampling_y = pd.DataFrame(self.sampling_y, columns=[self.target_variable])
       
        self.plot_target_variable(self.sampling_y)

    def sampling_process_ok(self):
        if self.sampling_x is not None:
            self.X_train = self.sampling_x
            self.y_train = self.sampling_y

        self.y_train = self.y_train.squeeze()
        self.clear_layout(self.preprocess_layout)
        self.clear_layout(self.preprocess_step_layout)
        QMessageBox.about(self, "complete", "Finished Preprocessing Step \n Now Go to the Modeling")
        self.model_setup()

    def model_setup(self):
        self.model_selection = QComboBox(self)
        self.model_selection.addItem('model selection')
        self.model_selection.addItem('Logistic Regression')
        self.model_selection.addItem('Decision Tree')
        self.model_selection.addItem('Random Forest')

        model_selection_layout = QHBoxLayout()
        model_selection_layout.addWidget(QLabel("Model Selection : "))
        model_selection_layout.addWidget(self.model_selection)
        self.model_setup_layout.addLayout(model_selection_layout)

        self.model_selection.activated[str].connect(self.hyperparameter_setup)

    def hyperparameter_setup(self, model_name):
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
            self.max_iters= QSpinBox()
            self.max_iters.setMinimum(1)
            self.max_iters.setMaximum(500)
            max_iters_layout = QHBoxLayout()
            max_iters_layout.addWidget(QLabel("Max Iterations : "))
            max_iters_layout.addWidget(self.max_iters)
            hyperparameter_layout.addLayout(max_iters_layout)

            self.model = LogisticRegression(max_iter=self.max_iters.value())
            self.max_iters.valueChanged.connect(self.logistic_model_update)

        elif model_name == 'Decision Tree':
            self.dt_max_depth = QSpinBox()
            self.dt_max_depth.setMinimum(1)
            max_depth_layout = QHBoxLayout()
            max_depth_layout.addWidget(QLabel('Max Depth : '))
            max_depth_layout.addWidget(self.dt_max_depth)
            hyperparameter_layout.addLayout(max_depth_layout)

            self.model = DecisionTreeClassifier(max_depth=self.dt_max_depth.value())
            self.dt_max_depth.valueChanged.connect(self.decision_tree_model_update)

        elif model_name == 'Random Forest':
            self.rf_max_depth = QSpinBox()
            self.rf_max_depth.setMinimum(1)
            max_depth_layout = QHBoxLayout()
            max_depth_layout.addWidget(QLabel('Max Depth : '))
            max_depth_layout.addWidget(self.rf_max_depth)
            hyperparameter_layout.addLayout(max_depth_layout)

            self.rf_max_depth.valueChanged.connect(self.random_forest_model_update)

            self.n_estimators = QSpinBox()
            self.n_estimators.setMinimum(1)
            self.n_estimators.setMaximum(150)
            n_estimators_layout = QHBoxLayout()
            n_estimators_layout.addWidget(QLabel('n_estimators : '))
            n_estimators_layout.addWidget(self.n_estimators)
            hyperparameter_layout.addLayout(n_estimators_layout)

            self.model = RandomForestClassifier(max_depth=self.rf_max_depth.value(), n_estimators=self.n_estimators.value())
            self.n_estimators.valueChanged.connect(self.random_forest_model_update)

    def logistic_model_update(self):
        self.model = LogisticRegression(max_iter=self.max_iters.value())

    def decision_tree_model_update(self):
        self.model = DecisionTreeClassifier(max_depth=self.dt_max_depth.value())

    def random_forest_model_update(self):
        self.model = RandomForestClassifier(max_depth=self.rf_max_depth.value(), n_estimators=self.n_estimators.value())

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred)
        cr = classification_report(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)

        QMessageBox.about(self, "accuracy score", str(accuracy))

        def plot_confusion_matrix(cm, classes):
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=classes, yticklabels=classes)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Confusion Matrix')
            plt.show()

        class_names = list(self.y_test.unique())
        plot_confusion_matrix(cm, class_names)

    def clear_layout(self, layout):
        # Iterate over all items in the layout and remove them
        while layout.count():
            item = layout.takeAt(0)  # Take item at index 0

            # If item is a widget, delete it
            if item.widget():
                item.widget().deleteLater()

            # If item is a layout, recursively clear it
            elif item.layout():
                self.clear_layout(item.layout())

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