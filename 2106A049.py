
import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QTabWidget, QPushButton, QLabel,
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit, QListWidget)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tkinter import ttk
import umap
from sklearn.manifold import TSNE
import pickle
import os

class NeuralNetDesigner(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neural Network Designer & Trainer")
        self.geometry("1000x700")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        # Variables
        self.layer_list = []
        self.model = None
        self.num_classes = 10  # default for MNIST
        self.input_shape = (28,28,1)  # default MNIST shape

        # Model Design Tab
        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="Model Design")

        # Training Tab
        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="Training")

        # GAN Tab
        self.gan_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.gan_tab, text="GAN")

        self.create_model_tab()
        self.create_train_tab()
        self.create_gan_tab()

    def create_model_tab(self):


        # Layer Type Selection
        tk.Label(self.model_tab, text="Layer Type:").grid(row=0, column=0, sticky="w")
        self.layer_type_var = tk.StringVar()
        self.layer_type_combo = ttk.Combobox(self.model_tab, textvariable=self.layer_type_var, state="readonly",
                                             values=['Dense', 'Conv2D', 'MaxPooling2D', 'Dropout', 'LSTM', 'GRU'])
        self.layer_type_combo.grid(row=0, column=1)
        self.layer_type_combo.bind("<<ComboboxSelected>>", self.update_param_fields)
        self.layer_type_combo.current(0)

        # Param fields frame
        self.params_frame = tk.Frame(self.model_tab)
        self.params_frame.grid(row=1, column=0, columnspan=3, sticky="w")

        # Add Layer Button
        self.add_layer_btn = tk.Button(self.model_tab, text="Add Layer", command=self.add_layer)
        self.add_layer_btn.grid(row=2, column=0, pady=5)

        # Layers Listbox
        self.layer_listbox = tk.Listbox(self.model_tab, height=10, width=50)
        self.layer_listbox.grid(row=3, column=0, columnspan=3, sticky="w")

        # Remove Layer Button
        self.remove_layer_btn = tk.Button(self.model_tab, text="Remove Selected Layer", command=self.remove_layer)
        self.remove_layer_btn.grid(row=4, column=0, pady=5)

        # Build Model Button
        self.build_model_btn = tk.Button(self.model_tab, text="Build Model", command=self.build_model)
        self.build_model_btn.grid(row=5, column=0, pady=5)

        # Model Summary Textbox
        self.summary_text = tk.Text(self.model_tab, height=15, width=80)
        self.summary_text.grid(row=6, column=0, columnspan=3, pady=10)

        # Pretrained Model Selection
        tk.Label(self.model_tab, text="Pretrained Model:").grid(row=11, column=0, sticky="w")
        self.pretrained_var = tk.StringVar()
        self.pretrained_combo = ttk.Combobox(self.model_tab, textvariable=self.pretrained_var, state="readonly")
        self.pretrained_combo['values'] = ('None', 'VGG16', 'ResNet50')
        self.pretrained_combo.current(0)
        self.pretrained_combo.grid(row=11, column=1)

        # Fine-tune checkbox
        self.finetune_var = tk.BooleanVar()
        tk.Checkbutton(self.model_tab, text="Fine-tune Pretrained", variable=self.finetune_var).grid(row=12, column=0, sticky="w")

        # Save/Load Model Buttons
        self.save_h5_btn = tk.Button(self.model_tab, text="Save Model (.h5)", command=self.save_model_h5)
        self.save_h5_btn.grid(row=9, column=0, pady=5)

        self.load_h5_btn = tk.Button(self.model_tab, text="Load Model (.h5)", command=self.load_model_h5)
        self.load_h5_btn.grid(row=9, column=1, pady=5)

        self.save_json_btn = tk.Button(self.model_tab, text="Save Architecture (.json)", command=self.save_model_json)
        self.save_json_btn.grid(row=10, column=0, pady=5)

        self.load_json_btn = tk.Button(self.model_tab, text="Load Architecture (.json)", command=self.load_model_json)
        self.load_json_btn.grid(row=10, column=1, pady=5)

        # Initialize params fields
        self.update_param_fields()


class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.regression_loss = None
        self.classification_loss = None
        self.current_loss = None
        self.current_loss_function = None
        self.current_model = None
        self.support_vectors = None
        
        # Neural network configuration
        self.layer_config = []
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()

    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()
            test_size = self.split_spin.value()

            if dataset_name == "Load Custom Dataset":
                self.load_custom_data()
                return

            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
                X = data.data
                y = data.target
                feature_names = data.feature_names

            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
                X = data.data
                y = data.target
                feature_names = data.feature_names

            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
                X = data.data
                y = data.target
                feature_names = [f"pixel_{i}" for i in range(X.shape[1])]

            elif dataset_name == "California Housing Dataset":
                data = fetch_california_housing()
                X = data.data
                y = data.target
                feature_names = data.feature_names

            elif dataset_name == "MNIST Dataset":
                (X, y), _ = tf.keras.datasets.mnist.load_data()
                X = X.reshape(X.shape[0], -1) / 255.0
                feature_names = [f"pixel_{i}" for i in range(X.shape[1])]

            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Apply scaling if selected
            self.apply_scaling()
            self.handle_missing_values()

            self.status_bar.showMessage(f"Loaded {dataset_name}")

            # Metrics güncelleme
            self.metrics_text.setText(
                f"Dataset Info:\n"
                f"Training samples: {self.X_train.shape[0]}\n"
                f"Test samples: {self.X_test.shape[0]}\n"
                f"Features: {self.X_train.shape[1]}\n"
                f"Feature names: {', '.join(feature_names)}"
            )

        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
    
    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            
            if file_name:
                # Load data
                data = pd.read_csv(file_name)
                
                # Ask user to select target column
                target_col = self.select_target_column(data.columns)
                
                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]
                    
                    # Split data
                    test_size = self.split_spin.value()
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X, y, 
                                                      test_size=test_size, 
                                                      random_state=42)
                    
                    # Apply scaling if selected
                    self.apply_scaling()
                    self.handle_missing_values()

                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
                    
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")
    
    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None
    
    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")
    
    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QHBoxLayout()

        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "California Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        
        # Data loading button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)

        # Missing values handling
        self.missing_values_combo = QComboBox()
        self.missing_values_combo.addItems([
            "No Handling",
            "Mean Imputation",
            "Interpolation",
            "Forward Fill",
            "Backward Fill"
        ])
        
        # Preprocessing options
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])
        
        # Train-test split options
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)
        
        # Add widgets to layout
        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(QLabel("Handling Missing Values:"))
        data_layout.addWidget(self.missing_values_combo)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        data_layout.addWidget(QLabel("Test Split:"))
        data_layout.addWidget(self.split_spin)
        
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)
    
    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab),
            ("Advanced Analysis", self.create_advanced_analysis_tab),
            ("GAN", self.create_gan_section),
            ("Logs", self.create_logging_section)
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)
    
    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()

        # Add loss function selection for regression
        loss_layout = QHBoxLayout()
        loss_layout.addWidget(QLabel("Loss Function:"))
        self.regression_loss_combo = QComboBox()
        self.regression_loss_combo.addItems([
            "Mean Squared Error (MSE)",
            "Mean Absolute Error (MAE)",
            "Huber Loss"
        ])
        loss_layout.addWidget(self.regression_loss_combo)
        regression_layout.addLayout(loss_layout)

        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox"}
        )
        regression_layout.addWidget(lr_group)

        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"]}
        )
        regression_layout.addWidget(logistic_group)

        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)

        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()

        # Add loss function selection for classification
        loss_layout = QHBoxLayout()
        loss_layout.addWidget(QLabel("Loss Function:"))
        self.classification_loss_combo = QComboBox()
        self.classification_loss_combo.addItems([
            "Cross-Entropy Loss",
            "Hinge Loss",
            "Squared Hinge Loss"
        ])
        loss_layout.addWidget(self.classification_loss_combo)
        classification_layout.addLayout(loss_layout)

        # Naive Bayes
        nb_group = self.create_algorithm_group(
            "Naive Bayes",
            {
                "var_smoothing": "double",
                "priors": {
                    "type": "text",
                    "placeholder": "Enter priors (e.g., 0.3,0.7)"
                }
            }
        )
        classification_layout.addWidget(nb_group)

        # SVM
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {
                "mode": ["SVC", "SVR"],
                "kernel": ["linear", "rbf", "polynomial"],
                "C": "double",
                "epsilon": "double",
                "degree": "int",
                "gamma": ["scale", "auto"],
                "coef0": "double"
            }
        )
        classification_layout.addWidget(svm_group)

        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]}
        )
        classification_layout.addWidget(dt_group)

        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"}
        )
        classification_layout.addWidget(rf_group)

        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        )
        classification_layout.addWidget(knn_group)

        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)

        return widget

    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # K-Means section
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()

        kmeans_params = self.create_algorithm_group(
            "K-Means Parameters",
            {"n_clusters": "int",
             "max_iter": "int",
             "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)

        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)

        # PCA section
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()

        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {"n_components": "int",
             "whiten": "checkbox"}
        )
        pca_layout.addWidget(pca_params)

        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)

        return widget

    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()

        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)

        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)

        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()

        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)

        return widget

    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)

        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)

        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)

    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()

        # Create parameter inputs
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))

            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            elif isinstance(param_type, dict) and param_type.get("type") == "text":
                widget = QLineEdit()
                if "placeholder" in param_type:
                    widget.setPlaceholderText(param_type["placeholder"])

                # Add tooltip for priors input
                if param_name == "priors":
                    widget.setToolTip("Enter comma-separated probabilities that sum to 1.0\n"
                                    "Example: 0.3,0.7 for binary classification")

            elif name == "Support Vector Machine" and param_name == "gamma":
                widget = QComboBox()
                gamma_options = ["scale", "auto"] + [str(0.1 * i) for i in range(1, 11)]
                widget.addItems(gamma_options)

            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)

        # Add train button
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)

        group.setLayout(layout)
        return group

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()

        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)

        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)

        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)

        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)

        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()

        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)

        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)

        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()

        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)

        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)

        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Model Architecture Section
        arch_group = QGroupBox("Model Architecture")
        arch_layout = QVBoxLayout()

        # Layer Configuration
        layer_list = QListWidget()
        self.layer_list = layer_list  # Save as instance variable
        arch_layout.addWidget(QLabel("Layers:"))
        arch_layout.addWidget(layer_list)

        # Layer Control Buttons
        btn_layout = QHBoxLayout()
        add_dense_btn = QPushButton("Add Dense Layer")
        add_conv_btn = QPushButton("Add Conv2D Layer")
        add_lstm_btn = QPushButton("Add LSTM Layer")
        remove_layer_btn = QPushButton("Remove Layer")

        btn_layout.addWidget(add_dense_btn)
        btn_layout.addWidget(add_conv_btn)
        btn_layout.addWidget(add_lstm_btn)
        btn_layout.addWidget(remove_layer_btn)

        add_dense_btn.clicked.connect(self.add_dense_layer_dialog)
        add_conv_btn.clicked.connect(self.add_conv_layer_dialog)
        add_lstm_btn.clicked.connect(self.add_lstm_layer_dialog)
        remove_layer_btn.clicked.connect(self.remove_selected_layer)

        arch_layout.addLayout(btn_layout)

        # Training Configuration
        train_config = QGroupBox("Training Configuration")
        train_layout = QGridLayout()

        # Optimizer Selection
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])
        train_layout.addWidget(QLabel("Optimizer:"), 0, 0)
        train_layout.addWidget(self.optimizer_combo, 0, 1)

        # Learning Rate
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 1.0)
        self.learning_rate.setValue(0.001)
        self.learning_rate.setSingleStep(0.0001)
        train_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        train_layout.addWidget(self.learning_rate, 1, 1)

        # Regularization
        self.dropout_rate = QDoubleSpinBox()
        self.dropout_rate.setRange(0.0, 0.9)
        self.dropout_rate.setValue(0.2)
        self.dropout_rate.setSingleStep(0.1)
        train_layout.addWidget(QLabel("Dropout Rate:"), 2, 0)
        train_layout.addWidget(self.dropout_rate, 2, 1)

        self.l2_lambda = QDoubleSpinBox()
        self.l2_lambda.setRange(0.0, 0.1)
        self.l2_lambda.setValue(0.01)
        self.l2_lambda.setSingleStep(0.001)
        train_layout.addWidget(QLabel("L2 Lambda:"), 3, 0)
        train_layout.addWidget(self.l2_lambda, 3, 1)

        train_config.setLayout(train_layout)

        # Model Operations
        model_ops = QGroupBox("Model Operations")
        model_ops_layout = QHBoxLayout()

        save_model_btn = QPushButton("Save Model")
        load_model_btn = QPushButton("Load Model")
        load_pretrained_btn = QPushButton("Load Pre-trained")

        model_ops_layout.addWidget(save_model_btn)
        model_ops_layout.addWidget(load_model_btn)
        model_ops_layout.addWidget(load_pretrained_btn)

        save_model_btn.clicked.connect(self.save_model)
        load_model_btn.clicked.connect(self.load_model)
        load_pretrained_btn.clicked.connect(self.load_pretrained_model)

        model_ops.setLayout(model_ops_layout)

        # Add all sections to main layout
        arch_group.setLayout(arch_layout)
        layout.addWidget(arch_group)
        layout.addWidget(train_config)
        layout.addWidget(model_ops)

        return widget

    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)

        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)

        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()

        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}

        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()

            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input

                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo

                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)

            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input

                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input

                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)

            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input

                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)

        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def add_layer():
            layer_type = type_combo.currentText()

            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))

            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })

            dialog.accept()

        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()

    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()

        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)

        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)

        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)

        group.setLayout(layout)
        return group

    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()

        # Placeholder for CNN-specific controls
        label = QLabel("CNN Controls (To be implemented)")
        layout.addWidget(label)

        group.setLayout(layout)
        return group

    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()

        # Placeholder for RNN-specific controls
        label = QLabel("RNN Controls (To be implemented)")
        layout.addWidget(label)

        group.setLayout(layout)
        return group

    def train_model(self, name, param_widgets):
        """Train selected model with parameters"""
        try:
            # Set loss function based on model type
            if name == "Linear Regression":
                if len(self.y_train.shape) > 1:
                    self.y_train = self.y_train.ravel()
                    self.y_test = self.y_test.ravel()

                params = {
                    'fit_intercept': param_widgets[
                        'fit_intercept'].isChecked() if 'fit_intercept' in param_widgets else True
                }

                self.current_model = LinearRegression(**params)
                self.current_model.fit(self.X_train, self.y_train)
                y_pred = self.current_model.predict(self.X_test)

                loss_name = self.regression_loss_combo.currentText()
                if loss_name == "Mean Squared Error (MSE)":
                    self.current_loss_function = mean_squared_error
                elif loss_name == "Mean Absolute Error (MAE)":
                    self.current_loss_function = lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
                elif loss_name == "Huber Loss":
                    delta = 1.0
                    self.current_loss_function = lambda y_true, y_pred: np.where(
                        np.abs(y_true - y_pred) <= delta,
                        0.5 * np.square(y_true - y_pred),
                        delta * np.abs(y_true - y_pred) - 0.5 * delta ** 2
                    ).mean()
                elif name in ["Logistic Regression", "Support Vector Machine", "Naive Bayes"]:
                    loss_name = self.classification_loss_combo.currentText()
                if loss_name == "Cross-Entropy Loss":
                    self.current_loss_function = lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred + 1e-10))
                elif loss_name == "Hinge Loss":
                    self.current_loss_function = lambda y_true, y_pred: np.mean(np.maximum(0, 1 - y_true * y_pred))
                elif loss_name == "Squared Hinge Loss":
                    self.current_loss_function = lambda y_true, y_pred: np.mean(np.square(np.maximum(0, 1 - y_true * y_pred)))

            # Get parameters from widgets
            params = {}
            for param_name, widget in param_widgets.items():
                if isinstance(widget, QSpinBox):
                    params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    params[param_name] = widget.currentText()
                elif isinstance(widget, QCheckBox):
                    params[param_name] = widget.isChecked()
                elif isinstance(widget, QLineEdit) and param_name == "priors":
                    priors_text = widget.text().strip()
                    if priors_text:
                        try:
                            # Parse comma-separated probabilities
                            priors = [float(p.strip()) for p in priors_text.split(',')]

                            # Validate probabilities
                            if abs(1.0 - sum(priors)) > 1e-6:
                                raise ValueError("Probabilities must sum to 1.0")

                            # Check if number of priors matches number of classes
                            if len(priors) != len(np.unique(self.y_train)):
                                raise ValueError("Number of priors must match number of classes")

                            params[param_name] = np.array(priors)
                        except Exception as e:
                            self.show_error(f"Invalid priors format: {str(e)}")
                            return

            # Train the model
            if name == "Linear Regression":
                self.current_model = LinearRegression(**params)
                self.current_model.fit(self.X_train, self.y_train)
                y_pred = self.current_model.predict(self.X_test)

            elif name == "Logistic Regression":
                self.current_model = LogisticRegression(**params)
                self.current_model.fit(self.X_train, self.y_train)
                y_pred = self.current_model.predict(self.X_test)

            elif name == "Naive Bayes":
                self.current_model = GaussianNB(**params)
                self.current_model.fit(self.X_train, self.y_train)
                y_pred = self.current_model.predict(self.X_test)

            elif name == "Support Vector Machine":
                # Get SVM parameters
                mode = params.get("mode", "SVC")
                kernel = params.get("kernel", "rbf")
                C = params.get("C", 1.0)
                epsilon = params.get("epsilon", 0.1)
                degree = params.get("degree", 3)
                gamma = params.get("gamma", "scale")
                coef0 = params.get("coef0", 0.0)

                # Create SVM model based on mode
                if mode == "SVC":
                    self.current_model = SVC(
                        kernel=kernel,
                        C=C,
                        degree=degree,
                        gamma=gamma,
                        coef0=coef0
                    )
                else:  # SVR mode
                    self.current_model = SVR(
                        kernel=kernel,
                        C=C,
                        epsilon=epsilon,
                        degree=degree,
                        gamma=gamma,
                        coef0=coef0
                    )

                self.current_model.fit(self.X_train, self.y_train)
                y_pred = self.current_model.predict(self.X_test)

                # Store additional info for visualization
                if hasattr(self.current_model, 'support_vectors_'):
                    self.support_vectors = self.current_model.support_vectors_

            elif name == "Decision Tree":
                self.current_model = DecisionTreeClassifier(max_depth=params["max_depth"],
                                                           min_samples_split=params["min_samples_split"],
                                                           criterion=params["criterion"])
                self.current_model.fit(self.X_train, self.y_train)
                y_pred = self.current_model.predict(self.X_test)

            elif name == "Random Forest":
                self.current_model = RandomForestClassifier(n_estimators=params["n_estimators"],
                                                           max_depth=params["max_depth"],
                                                           min_samples_split=params["min_samples_split"])
                self.current_model.fit(self.X_train, self.y_train)
                y_pred = self.current_model.predict(self.X_test)

            elif name == "K-Nearest Neighbors":
                self.current_model = KNeighborsClassifier(n_neighbors=params["n_neighbors"],
                                                         weights=params["weights"],
                                                         metric=params["metric"])
                self.current_model.fit(self.X_train, self.y_train)
                y_pred = self.current_model.predict(self.X_test)

            # Update visualization and metrics
            self.update_visualization(y_pred)
            self.update_metrics(y_pred)

            self.status_bar.showMessage(f"{name} training complete")

        except Exception as e:
            self.show_error(f"Error training model: {str(e)}")

    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return

        try:
            # Create and compile model
            model = self.create_neural_network()

            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()

            # Prepare data for neural network
            if len(self.X_train.shape) == 1:
                X_train = self.X_train.reshape(-1, 1)
                X_test = self.X_test.reshape(-1, 1)
            else:
                X_train = self.X_train
                X_test = self.X_test

            # One-hot encode target for classification
            y_train = tf.keras.utils.to_categorical(self.y_train)
            y_test = tf.keras.utils.to_categorical(self.y_test)

            # Compile model
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # Train model
            history = model.fit(X_train, y_train,
                               batch_size=batch_size,
                               epochs=epochs,
                               validation_data=(X_test, y_test),
                               callbacks=[self.create_progress_callback()])

            # Update visualization with training history
            self.plot_training_history(history)

            self.status_bar.showMessage("Neural Network Training Complete")

        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")

    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()

        # Add layers based on configuration
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]

            if layer_type == "Dense":
                model.add(layers.Dense(**params))
            elif layer_type == "Conv2D":
                # Add input shape for the first layer
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.Conv2D(**params))
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))

        # Add output layer based on number of classes
        num_classes = len(np.unique(self.y_train))
        model.add(layers.Dense(num_classes, activation='softmax'))

        return model

    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar

            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)

        return ProgressCallback(self.progress_bar)

    def update_visualization(self, y_pred):
        """Update the visualization with current results"""
        self.figure.clear()

        # Create appropriate visualization based on data
        if len(np.unique(self.y_test)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()],
                   [self.y_test.min(), self.y_test.max()],
                   'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")

        else:  # Classification
            if self.X_train.shape[1] > 2:  # Use PCA for visualization
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)

                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)

            else:  # Direct 2D visualization
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)

        # Special visualization for SVM
        if isinstance(self.current_model, (SVC, SVR)) and self.X_test.shape[1] == 2:
            ax = self.figure.add_subplot(111)

            # Plot decision boundary for classification
            if isinstance(self.current_model, SVC):
                x_min, x_max = self.X_test[:, 0].min() - 1, self.X_test[:, 0].max() + 1
                y_min, y_max = self.X_test[:, 1].min() - 1, self.X_test[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                   np.arange(y_min, y_max, 0.02))
                Z = self.current_model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, alpha=0.4)

            # Plot data points
            scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                               c=y_pred, cmap='viridis')

            # Plot support vectors if available
            if hasattr(self, 'support_vectors'):
                ax.scatter(self.support_vectors[:, 0], self.support_vectors[:, 1],
                          s=100, linewidth=1, facecolors='none', edgecolors='r',
                          label='Support Vectors')
                ax.legend()

            return

        self.canvas.draw()

    def update_metrics(self, y_pred):
        """Update metrics display"""
        metrics_text = "Model Performance Metrics:\n\n"

        # Check if current model is Linear Regression
        if isinstance(self.current_model, LinearRegression):
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(self.y_test - y_pred))
            r2 = self.current_model.score(self.X_test, self.y_test)

            metrics_text = "Model Performance Metrics:\n\n"
            metrics_text += f"Mean Squared Error: {mse:.4f}\n"
            metrics_text += f"Root Mean Squared Error: {rmse:.4f}\n"
            metrics_text += f"Mean Absolute Error: {mae:.4f}\n"
            metrics_text += f"R² Score: {r2:.4f}"

            if hasattr(self, 'current_loss_function'):
                current_loss = self.regression_loss_combo.currentText()
                loss_value = self.current_loss_function(self.y_test, y_pred)
                metrics_text += f"\n\nSelected Loss ({current_loss}): {loss_value:.4f}"

            self.metrics_text.setText(metrics_text)
        
        elif len(np.unique(self.y_test)) <= 10:  # Classification
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(conf_matrix)

        self.metrics_text.setText(metrics_text)

    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()

        # Plot training & validation accuracy
        ax1 = self.figure.add_subplot(211)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Test'])

        # Plot training & validation loss
        ax2 = self.figure.add_subplot(212)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Test'])

        self.figure.tight_layout()
        self.canvas.draw()

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

    def handle_missing_values(self):
        """Apply selected missing data handling method."""
        method = self.missing_values_combo.currentText()

        # Controlling data load part
        if self.X_train is None or self.X_test is None:
            return

        # Selecting methods part
        if method == "No Handling":
            return

        elif method == "Mean Imputation":
            imputer = SimpleImputer(strategy="mean")

        elif method == "Interpolation":
            self.X_train = pd.DataFrame(self.X_train).interpolate().values
            self.X_test = pd.DataFrame(self.X_test).interpolate().values
            return

        elif method == "Forward Fill":
            self.X_train = pd.DataFrame(self.X_train).fillna(method="ffill").values
            self.X_test = pd.DataFrame(self.X_test).fillna(method="ffill").values
            return

        elif method == "Backward Fill":
            self.X_train = pd.DataFrame(self.X_train).fillna(method="bfill").values
            self.X_test = pd.DataFrame(self.X_test).fillna(method="bfill").values
            return

        # Processing part
        self.X_train = imputer.fit_transform(self.X_train)
        self.X_test = imputer.transform(self.X_test)

    def create_advanced_analysis_tab(self):
        """advanced analysis tab with PCA, t-SNE, UMAP and K-fold cross validation"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Dimensionality Reduction Section
        dim_group = QGroupBox("Dimensionality Reduction")
        dim_layout = QGridLayout()

        # PCA Controls
        self.pca_components = QSpinBox()
        self.pca_components.setRange(1, 10)
        self.pca_components.setValue(2)

        self.variance_threshold = QDoubleSpinBox()
        self.variance_threshold.setRange(0.1, 0.99)
        self.variance_threshold.setValue(0.95)
        self.variance_threshold.setSingleStep(0.05)

        dim_layout.addWidget(QLabel("PCA Components:"), 0, 0)
        dim_layout.addWidget(self.pca_components, 0, 1)
        dim_layout.addWidget(QLabel("Variance Threshold:"), 1, 0)
        dim_layout.addWidget(self.variance_threshold, 1, 1)

        # t-SNE Controls
        self.perplexity = QDoubleSpinBox()
        self.perplexity.setRange(5.0, 50.0)
        self.perplexity.setValue(30.0)
        dim_layout.addWidget(QLabel("t-SNE Perplexity:"), 2, 0)
        dim_layout.addWidget(self.perplexity, 2, 1)

        # UMAP Controls
        self.n_neighbors = QSpinBox()
        self.n_neighbors.setRange(2, 100)
        self.n_neighbors.setValue(15)
        dim_layout.addWidget(QLabel("UMAP Neighbors:"), 3, 0)
        dim_layout.addWidget(self.n_neighbors, 3, 1)

        self.run_pca_btn = QPushButton("Run PCA")
        self.run_tsne_btn = QPushButton("Run t-SNE")
        self.run_umap_btn = QPushButton("Run UMAP")

        dim_layout.addWidget(self.run_pca_btn, 4, 0)
        dim_layout.addWidget(self.run_tsne_btn, 4, 1)
        dim_layout.addWidget(self.run_umap_btn, 4, 2)

        dim_group.setLayout(dim_layout)
        layout.addWidget(dim_group)

        # Cross Validation Section
        cv_group = QGroupBox("Cross Validation")
        cv_layout = QGridLayout()

        # K-Fold Controls
        self.k_folds = QSpinBox()
        self.k_folds.setRange(2, 10)
        self.k_folds.setValue(5)

        cv_layout.addWidget(QLabel("K-Folds:"), 0, 0)
        cv_layout.addWidget(self.k_folds, 0, 1)

        # Split Ratio Controls
        self.train_ratio = QDoubleSpinBox()
        self.train_ratio.setRange(0.5, 0.9)
        self.train_ratio.setValue(0.7)
        self.train_ratio.setSingleStep(0.05)

        cv_layout.addWidget(QLabel("Train Ratio:"), 1, 0)
        cv_layout.addWidget(self.train_ratio, 1, 1)

        self.run_cv_btn = QPushButton("Run Cross Validation")
        cv_layout.addWidget(self.run_cv_btn, 2, 0, 1, 2)

        cv_group.setLayout(cv_layout)
        layout.addWidget(cv_group)

        self.run_pca_btn.clicked.connect(self.run_pca_analysis)
        self.run_tsne_btn.clicked.connect(self.run_tsne_analysis)
        self.run_umap_btn.clicked.connect(self.run_umap_analysis)
        self.run_cv_btn.clicked.connect(self.run_cross_validation)

        return widget

    def run_pca_analysis(self):
        """PCA analysis function"""
        try:
            n_components = self.pca_components.value()
            variance_threshold = self.variance_threshold.value()

            # PCA thing
            pca = PCA()
            X_pca = pca.fit_transform(self.X_train)

            # Calculation
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

            # Graph
            self.figure.clear()
            ax1 = self.figure.add_subplot(211)
            ax1.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
            ax1.axhline(y=variance_threshold, color='r', linestyle='--')
            ax1.set_xlabel('Component Amount')
            ax1.set_ylabel('Cumulative Total Exp. Var.')
            ax1.set_title('PCA Variance Analysis')

            if X_pca.shape[1] >= 2:
                ax2 = self.figure.add_subplot(212)
                scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_train)
                ax2.set_xlabel('Primary Component')
                ax2.set_ylabel('Secondary Component')
                ax2.set_title('2D PCA Graph')
                self.figure.colorbar(scatter)

            self.figure.tight_layout()
            self.canvas.draw()

            metrics_text = "PCA Analysis Result:\n\n"
            for i, ratio in enumerate(explained_variance_ratio[:n_components]):
                metrics_text += f"Components {i + 1}: {ratio:.4f}\n"
            metrics_text += f"\nTotal Explained Variance: {sum(explained_variance_ratio[:n_components]):.4f}"
            self.metrics_text.setText(metrics_text)

        except Exception as e:
            self.show_error(f"PCA Analysis Error!: {str(e)}")

    def run_tsne_analysis(self):
        """t-SNE Analysis"""
        try:
            perplexity = self.perplexity.value()

            # t-SNE thing
            tsne = TSNE(n_components=2, perplexity=perplexity)
            X_tsne = tsne.fit_transform(self.X_train)

            # Graph
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.y_train)
            ax.set_title('t-SNE Graph')
            self.figure.colorbar(scatter)
            self.canvas.draw()

        except Exception as e:
            self.show_error(f"t-SNE Analysis Error!: {str(e)}")

    def run_umap_analysis(self):
        """UMAP Analysis"""
        try:
            n_neighbors = self.n_neighbors.value()

            # UMAP thing
            reducer = umap.UMAP(n_neighbors=n_neighbors)
            X_umap = reducer.fit_transform(self.X_train)

            # Graph
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=self.y_train)
            ax.set_title('UMAP Graph')
            self.figure.colorbar(scatter)
            self.canvas.draw()

        except Exception as e:
            self.show_error(f"UMAP Analysis Error!: {str(e)}")

    def run_cross_validation(self):
        """K-fold cross validation"""
        try:
            from sklearn.model_selection import KFold
            k_folds = self.k_folds.value()

            if self.current_model is None:
                self.show_error("Please train a model first at 'Classical ML Tab'.")
                return

            # K-fold CV thing
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
                X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
                y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]

                self.current_model.fit(X_fold_train, y_fold_train)
                score = self.current_model.score(X_fold_val, y_fold_val)
                scores.append(score)

            # New Results
            metrics_text = "Cross Validation Results:\n\n"
            for i, score in enumerate(scores):
                metrics_text += f"Fold {i + 1}: {score:.4f}\n"
            metrics_text += f"\nMean: {np.mean(scores):.4f}"
            metrics_text += f"\nStandard Deviation: {np.std(scores):.4f}"
            self.metrics_text.setText(metrics_text)

        except Exception as e:
            self.show_error(f"Cross Validation Error!: {str(e)}")

    def add_dense_layer_dialog(self):
        """Dialog for adding a dense layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Dense Layer")
        layout = QVBoxLayout(dialog)

        # Units
        units_layout = QHBoxLayout()
        units_layout.addWidget(QLabel("Units:"))
        units_spin = QSpinBox()
        units_spin.setRange(1, 1024)
        units_spin.setValue(64)
        units_layout.addWidget(units_spin)
        layout.addLayout(units_layout)

        # Activation
        activation_layout = QHBoxLayout()
        activation_layout.addWidget(QLabel("Activation:"))
        activation_combo = QComboBox()
        activation_combo.addItems(["relu", "sigmoid", "tanh"])
        activation_layout.addWidget(activation_combo)
        layout.addLayout(activation_layout)

        # Add button
        add_btn = QPushButton("Add Layer")
        add_btn.clicked.connect(lambda: self.add_layer_to_list(
            f"Dense({units_spin.value()}, {activation_combo.currentText()})",
            {"type": "Dense", "units": units_spin.value(),
             "activation": activation_combo.currentText()}))
        add_btn.clicked.connect(dialog.accept)
        layout.addWidget(add_btn)

        dialog.exec()

    def add_conv_layer_dialog(self):
        """Dialog for adding a convolutional layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Convolutional Layer")
        layout = QVBoxLayout(dialog)

        # Filters
        filters_layout = QHBoxLayout()
        filters_layout.addWidget(QLabel("Filters:"))
        filters_spin = QSpinBox()
        filters_spin.setRange(1, 512)
        filters_spin.setValue(32)
        filters_layout.addWidget(filters_spin)
        layout.addLayout(filters_layout)

        # Kernel Size
        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("Kernel Size:"))
        kernel_spin = QSpinBox()
        kernel_spin.setRange(1, 7)
        kernel_spin.setValue(3)
        kernel_layout.addWidget(kernel_spin)
        layout.addLayout(kernel_layout)

        # Add button
        add_btn = QPushButton("Add Layer")
        add_btn.clicked.connect(lambda: self.add_layer_to_list(
            f"Conv2D({filters_spin.value()}, {kernel_spin.value()}x{kernel_spin.value()})",
            {"type": "Conv2D", "filters": filters_spin.value(),
             "kernel_size": (kernel_spin.value(), kernel_spin.value())}))
        add_btn.clicked.connect(dialog.accept)
        layout.addWidget(add_btn)
        add_btn.clicked.connect(lambda: self.add_layer_to_list(
            f"Conv2D({filters_spin.value()}, {kernel_spin.value()}x{kernel_spin.value()})",
            {
                "type": "Conv2D",
                "filters": filters_spin.value(),
                "kernel_size": kernel_spin.value(),
                "activation": "relu"
            }))

        dialog.exec()

    def add_layer_to_list(self, layer_text, layer_config):
        """Add a layer to the layer list"""
        self.layer_list.addItem(layer_text)
        if not hasattr(self, 'layer_configs'):
            self.layer_configs = []
        self.layer_configs.append({
            "type": layer_config.get("type", "Dense"),
            "units": layer_config.get("units", 64),
            "activation": layer_config.get("activation", "relu"),
            "filters": layer_config.get("filters", None),
            "kernel_size": layer_config.get("kernel_size", None),
            "return_sequences": layer_config.get("return_sequences", False)
        })

    def remove_selected_layer(self):
        """Remove the selected layer from the list"""
        current_row = self.layer_list.currentRow()
        if current_row >= 0:
            self.layer_list.takeItem(current_row)
            if hasattr(self, 'layer_configs'):
                self.layer_configs.pop(current_row)

    def save_model(self):
        """Save the current model architecture"""
        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Model", "", "HDF5 files (*.h5);;JSON files (*.json)")

            if file_name:
                if file_name.endswith('.h5'):
                    if isinstance(self.current_model, tf.keras.Model):
                        self.current_model.save(file_name)
                    else:
                        with open(file_name, 'wb') as f:
                            pickle.dump(self.current_model, f)

                elif file_name.endswith('.json'):
                    if isinstance(self.current_model, tf.keras.Model):
                        model_json = self.current_model.to_json()
                        with open(file_name, "w") as json_file:
                            json_file.write(model_json)
                        # Ağırlıkları ayrı kaydet
                        weights_file = file_name.replace('.json', '_weights.h5')
                        self.current_model.save_weights(weights_file)
                    else:
                        self.show_error("JSON format only supports Keras models")
                        return

                self.status_bar.showMessage(f"Model saved to {file_name}")

        except Exception as e:
            self.show_error(f"Error saving model: {str(e)}")

    def load_model(self):
        """Load a saved model"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Load Model", "", "HDF5 files (*.h5);;JSON files (*.json)")

            if file_name:
                if file_name.endswith('.h5'):
                    try:
                        self.current_model = tf.keras.models.load_model(file_name)
                    except:
                        with open(file_name, 'rb') as f:
                            self.current_model = pickle.load(f)

                elif file_name.endswith('.json'):

                    with open(file_name, 'r') as json_file:
                        model_json = json_file.read()
                    self.current_model = tf.keras.models.model_from_json(model_json)

                    weights_file = file_name.replace('.json', '_weights.h5')
                    if os.path.exists(weights_file):
                        self.current_model.load_weights(weights_file)

                if isinstance(self.current_model, tf.keras.Model):
                    model_summary = []
                    self.current_model.summary(print_fn=lambda x: model_summary.append(x))
                    self.metrics_text.setText('\n'.join(model_summary))
                else:
                    self.metrics_text.setText(f"Loaded model type: {type(self.current_model).__name__}")

                if isinstance(self.current_model, tf.keras.Model):
                    self.layer_list.clear()
                    for layer in self.current_model.layers:
                        self.layer_list.addItem(f"{layer.name} ({layer.__class__.__name__})")

                self.status_bar.showMessage(f"Model loaded from {file_name}")

        except Exception as e:
            self.show_error(f"Error loading model: {str(e)}")

    def add_lstm_layer_dialog(self):
        """Dialog for adding an LSTM layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add LSTM Layer")
        layout = QVBoxLayout(dialog)

        # Units
        units_layout = QHBoxLayout()
        units_layout.addWidget(QLabel("Units:"))
        units_spin = QSpinBox()
        units_spin.setRange(1, 512)
        units_spin.setValue(64)
        units_layout.addWidget(units_spin)
        layout.addLayout(units_layout)

        # Return Sequences
        return_seq_layout = QHBoxLayout()
        return_seq_layout.addWidget(QLabel("Return Sequences:"))
        return_seq_check = QCheckBox()
        return_seq_layout.addWidget(return_seq_check)
        layout.addLayout(return_seq_layout)

        # Activation
        activation_layout = QHBoxLayout()
        activation_layout.addWidget(QLabel("Activation:"))
        activation_combo = QComboBox()
        activation_combo.addItems(["tanh", "relu", "sigmoid"])
        activation_layout.addWidget(activation_combo)
        layout.addLayout(activation_layout)

        # Add button
        add_btn = QPushButton("Add Layer")
        add_btn.clicked.connect(lambda: self.add_layer_to_list(
            f"LSTM({units_spin.value()}, return_sequences={return_seq_check.isChecked()})",
            {"type": "LSTM",
             "units": units_spin.value(),
             "return_sequences": return_seq_check.isChecked(),
             "activation": activation_combo.currentText()}))
        add_btn.clicked.connect(dialog.accept)
        layout.addWidget(add_btn)

        dialog.exec()

    def load_pretrained_model(self):
        """Load a pre-trained model (VGG16, ResNet, etc.)"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Load Pre-trained Model")
        layout = QVBoxLayout(dialog)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Select Model:"))
        model_combo = QComboBox()
        model_combo.addItems(["VGG16", "ResNet50", "MobileNet", "DenseNet121"])
        model_layout.addWidget(model_combo)
        layout.addLayout(model_layout)

        # Include top layer option
        include_top_layout = QHBoxLayout()
        include_top_layout.addWidget(QLabel("Include Top Layer:"))
        include_top_check = QCheckBox()
        include_top_check.setChecked(False)
        include_top_layout.addWidget(include_top_check)
        layout.addLayout(include_top_layout)

        # Load button
        load_btn = QPushButton("Load Model")

        def load_selected_model():
            try:
                model_name = model_combo.currentText()
                include_top = include_top_check.isChecked()

                if model_name == "VGG16":
                    self.current_model = tf.keras.applications.VGG16(
                        include_top=include_top,
                        weights='imagenet',
                        input_shape=(224, 224, 3)
                    )
                elif model_name == "ResNet50":
                    self.current_model = tf.keras.applications.ResNet50(
                        include_top=include_top,
                        weights='imagenet',
                        input_shape=(224, 224, 3)
                    )
                elif model_name == "MobileNet":
                    self.current_model = tf.keras.applications.MobileNet(
                        include_top=include_top,
                        weights='imagenet',
                        input_shape=(224, 224, 3)
                    )
                elif model_name == "DenseNet121":
                    self.current_model = tf.keras.applications.DenseNet121(
                        include_top=include_top,
                        weights='imagenet',
                        input_shape=(224, 224, 3)
                    )

                model_summary = []
                self.current_model.summary(print_fn=lambda x: model_summary.append(x))
                self.metrics_text.setText('\n'.join(model_summary))

                self.layer_list.clear()
                for layer in self.current_model.layers:
                    self.layer_list.addItem(f"{layer.name} ({layer.__class__.__name__})")

                self.status_bar.showMessage(f"Loaded pre-trained {model_name}")
                dialog.accept()

            except Exception as e:
                self.show_error(f"Error loading pre-trained model: {str(e)}")

        load_btn.clicked.connect(load_selected_model)
        layout.addWidget(load_btn)

        dialog.exec()

    class TrainingCallback(tf.keras.callbacks.Callback):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent
            self.weight_history = []
            self.gradient_history = []

        def on_epoch_end(self, epoch, logs=None):
            if logs:
                self.parent.update_training_plots(logs)

            weights = []
            gradients = []

            test_data = self.parent.X_test[:100]
            with tf.GradientTape() as tape:
                predictions = self.model(test_data)
                loss = self.model.loss(self.parent.y_test[:100], predictions)

            grads = tape.gradient(loss, self.model.trainable_weights)

            for w, g in zip(self.model.trainable_weights, grads):
                weights.append(w.numpy().flatten())
                gradients.append(g.numpy().flatten())

            self.weight_history.append(np.concatenate(weights))
            self.gradient_history.append(np.concatenate(gradients))

            # Görselleştirmeyi güncelle
            self.parent.plot_weight_gradients(self.weight_history, self.gradient_history)

    def plot_weight_gradients(self, weight_history, gradient_history):
        if not weight_history or not gradient_history:
            return

        self.figure.clear()

        ax1 = self.figure.add_subplot(211)
        ax1.hist(weight_history[-1], bins=50, alpha=0.7)
        ax1.set_title("Weight Distribution")
        ax1.set_xlabel("Weight Value")
        ax1.set_ylabel("Frequency")

        ax2 = self.figure.add_subplot(212)
        ax2.hist(gradient_history[-1], bins=50, alpha=0.7)
        ax2.set_title("Gradient Distribution")
        ax2.set_xlabel("Gradient Value")
        ax2.set_ylabel("Frequency")

        self.figure.tight_layout()
        self.canvas.draw()
    def create_data_augmentation(self):
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1)
        ])

    def fine_tune_model(self):
        for layer in self.current_model.layers[:-3]:
            layer.trainable = False

        model = tf.keras.Sequential([
            self.current_model,
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

    def compile_model(self):
        if self.optimizer_combo.currentText() == "Adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate.value()
            )
        elif self.optimizer_combo.currentText() == "SGD":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate.value()
            )

        regularizer = tf.keras.regularizers.l2(self.l2_lambda.value())

    def create_callbacks(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='best_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            self.create_progress_callback()
        ]

    def update_layer_list(self):
        self.layer_list.clear()
        for idx, layer_config in enumerate(self.layer_configs):
            layer_type = layer_config["type"]
            if layer_type == "Dense":
                text = f"{idx}: Dense(units={layer_config['units']}, activation={layer_config['activation']})"
            elif layer_type == "Conv2D":
                text = f"{idx}: Conv2D(filters={layer_config['filters']}, kernel={layer_config['kernel_size']})"
            elif layer_type == "LSTM":
                text = f"{idx}: LSTM(units={layer_config['units']}, return_seq={layer_config['return_sequences']})"
            self.layer_list.addItem(text)

    def edit_layer_dialog(self):
        current_row = self.layer_list.currentRow()
        if current_row < 0:
            return

        layer_config = self.layer_configs[current_row]
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit {layer_config['type']} Layer")
        layout = QVBoxLayout(dialog)

        param_widgets = {}
        if layer_config["type"] == "Dense":
            units_layout = QHBoxLayout()
            units_layout.addWidget(QLabel("Units:"))
            units_spin = QSpinBox()
            units_spin.setRange(1, 1024)
            units_spin.setValue(layer_config["units"])
            units_layout.addWidget(units_spin)
            layout.addLayout(units_layout)
            param_widgets["units"] = units_spin

        elif layer_config["type"] == "Conv2D":
            filters_layout = QHBoxLayout()
            filters_layout.addWidget(QLabel("Filters:"))
            filters_spin = QSpinBox()
            filters_spin.setRange(1, 512)
            filters_spin.setValue(layer_config["filters"])
            filters_layout.addWidget(filters_spin)
            layout.addLayout(filters_layout)
            param_widgets["filters"] = filters_spin

        # Activation for all layers
        activation_layout = QHBoxLayout()
        activation_layout.addWidget(QLabel("Activation:"))
        activation_combo = QComboBox()
        activation_combo.addItems(["relu", "sigmoid", "tanh"])
        activation_combo.setCurrentText(layer_config["activation"])
        activation_layout.addWidget(activation_combo)
        layout.addLayout(activation_layout)
        param_widgets["activation"] = activation_combo

        def save_changes():
            for param, widget in param_widgets.items():
                if isinstance(widget, QSpinBox):
                    layer_config[param] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_config[param] = widget.currentText()
            self.update_layer_list()
            dialog.accept()

        save_btn = QPushButton("Save Changes")
        save_btn.clicked.connect(save_changes)
        layout.addWidget(save_btn)

        dialog.exec()

    def setup_tensorboard(self):
        """TensorBoard setup function"""
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'logs/{current_time}'
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )

    def create_logging_section(self):
        """Loging function"""
        log_group = QGroupBox("Training Logs")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)

        tensorboard_btn = QPushButton("Launch TensorBoard")
        tensorboard_btn.clicked.connect(self.launch_tensorboard)
        log_layout.addWidget(tensorboard_btn)

        log_group.setLayout(log_layout)
        return log_group

    def launch_tensorboard(self):
        """TensorBoard function"""
        try:
            import webbrowser
            import subprocess
            subprocess.Popen(["tensorboard", "--logdir", "logs/"])
            webbrowser.open("http://localhost:6006")
        except Exception as e:
            self.show_error(f"TensorBoard launch error: {str(e)}")

    def create_gan_section(self):
        """GAN controls"""
        gan_group = QGroupBox("GAN Training")
        gan_layout = QVBoxLayout()

        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.gan_batch_size = QSpinBox()
        self.gan_batch_size.setRange(16, 256)
        self.gan_batch_size.setValue(32)
        self.gan_batch_size.setSingleStep(16)
        batch_layout.addWidget(self.gan_batch_size)
        gan_layout.addLayout(batch_layout)

        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("Epochs:"))
        self.gan_epochs = QSpinBox()
        self.gan_epochs.setRange(100, 10000)
        self.gan_epochs.setValue(1000)
        self.gan_epochs.setSingleStep(100)
        epoch_layout.addWidget(self.gan_epochs)
        gan_layout.addLayout(epoch_layout)

        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.gan_lr = QDoubleSpinBox()
        self.gan_lr.setRange(0.0001, 0.01)
        self.gan_lr.setValue(0.0002)
        self.gan_lr.setSingleStep(0.0001)
        lr_layout.addWidget(self.gan_lr)
        gan_layout.addLayout(lr_layout)

        gen_layout = QHBoxLayout()
        gen_layout.addWidget(QLabel("Generator Layers:"))
        self.gen_layers = QSpinBox()
        self.gen_layers.setRange(2, 10)
        self.gen_layers.setValue(4)
        gen_layout.addWidget(self.gen_layers)
        gan_layout.addLayout(gen_layout)

        disc_layout = QHBoxLayout()
        disc_layout.addWidget(QLabel("Discriminator Layers:"))
        self.disc_layers = QSpinBox()
        self.disc_layers.setRange(2, 10)
        self.disc_layers.setValue(3)
        disc_layout.addWidget(self.disc_layers)
        gan_layout.addLayout(disc_layout)

        latent_layout = QHBoxLayout()
        latent_layout.addWidget(QLabel("Latent Dimension:"))
        self.latent_dim = QSpinBox()
        self.latent_dim.setRange(10, 200)
        self.latent_dim.setValue(100)
        latent_layout.addWidget(self.latent_dim)
        gan_layout.addLayout(latent_layout)

        train_gan_btn = QPushButton("Train GAN")
        train_gan_btn.clicked.connect(self.train_gan)
        gan_layout.addWidget(train_gan_btn)

        generate_btn = QPushButton("Generate Samples")
        generate_btn.clicked.connect(self.generate_samples)
        gan_layout.addWidget(generate_btn)

        gan_group.setLayout(gan_layout)
        return gan_group

    def build_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(256, input_dim=self.latent_dim.value()),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Dense(1024),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Dense(np.prod(self.X_train.shape[1:]), activation='tanh')
        ])
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            layers.Dense(1024, input_shape=(np.prod(self.X_train.shape[1:]),)),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def train_gan(self):
        """Train GAN model"""
        try:
            # Create models
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()

            batch_size = self.gan_batch_size.value()
            epochs = self.gan_epochs.value()
            learning_rate = self.gan_lr.value()

            d_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
            g_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)

            # Compile discriminator
            self.discriminator.compile(
                optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # Create GAN model
            gan_input = tf.keras.Input(shape=(self.latent_dim.value(),))
            x = self.generator(gan_input)
            self.discriminator.trainable = False
            gan_output = self.discriminator(x)
            self.combined = tf.keras.Model(gan_input, gan_output)

            # Compile GAN
            self.combined.compile(
                optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                loss='binary_crossentropy'
            )

            # Training parameters
            batch_size = 32
            epochs = 1000

            # Set progress bar
            self.progress_bar.setMaximum(epochs)

            # Training loop
            for epoch in range(epochs):
                # Get batch from real data
                idx = np.random.randint(0, self.X_train.shape[0], batch_size)
                real_data = self.X_train[idx]

                # Generate noise and fake data
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim.value()))
                fake_data = self.generator.predict(noise)

                # Train discriminator
                d_loss_real = self.discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
                d_loss_fake = self.discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train generator
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim.value()))
                g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

                # Update progress bar and metrics
                self.progress_bar.setValue(epoch + 1)
                self.metrics_text.setText(f"Epoch {epoch + 1}/{epochs}\n"
                                          f"D loss: {d_loss[0]:.4f}\n"
                                          f"G loss: {g_loss:.4f}")

                # Generate samples every 100 epochs
                if epoch % 100 == 0:
                    self.generate_samples()

                # GUI update
                QApplication.processEvents()

            self.status_bar.showMessage("GAN training completed")

        except Exception as e:
            self.show_error(f"GAN training error: {str(e)}")

    def generate_samples(self):
        """Generate samples with GAN"""
        try:
            if not hasattr(self, 'generator'):
                self.show_error("Generator model not found. Train GAN first.")
                return

            # Generate random noise
            noise = np.random.normal(0, 1, (16, self.latent_dim.value()))

            # Generate samples
            generated_samples = self.generator.predict(noise)

            # Visualize
            self.figure.clear()
            rows, cols = 4, 4

            # Check data dimensions
            if self.X_train.shape[1] == 1:
                ax = self.figure.add_subplot(111)
                ax.hist(generated_samples, bins=30, alpha=0.7, label='Generated')
                ax.hist(self.X_train, bins=30, alpha=0.7, label='Real')
                ax.legend()
                ax.set_title('Generated vs Real Data Distribution')

            elif self.X_train.shape[1] == 2:
                ax = self.figure.add_subplot(111)
                ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c='blue', alpha=0.5, label='Real')
                ax.scatter(generated_samples[:, 0], generated_samples[:, 1], c='red', alpha=0.5, label='Generated')
                ax.legend()
                ax.set_title('Generated vs Real Data')

            else:
                ax = self.figure.add_subplot(111)
                ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c='blue', alpha=0.5, label='Real')
                ax.scatter(generated_samples[:, 0], generated_samples[:, 1], c='red', alpha=0.5, label='Generated')
                ax.legend()
                ax.set_title('Generated vs Real Data (First 2 Dimensions)')

            self.figure.tight_layout()
            self.canvas.draw()
            self.status_bar.showMessage("Samples generated")

        except Exception as e:
            self.show_error(f"Sample generation error: {str(e)}")

    def create_lr_scheduler(self):
        """Creating learning rate scheduler function"""
        scheduler_type = self.scheduler_combo.currentText()
        initial_lr = self.learning_rate.value()

        if scheduler_type == "Step Decay":
            decay_rate = self.decay_rate.value()
            decay_steps = self.decay_steps.value()
            return tf.keras.optimizers.schedules.StepDecay(
                initial_learning_rate=initial_lr,
                decay_steps=decay_steps,
                decay_rate=decay_rate
            )

        elif scheduler_type == "Exponential":
            decay_rate = self.decay_rate.value()
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_lr,
                decay_steps=100,
                decay_rate=decay_rate
            )

        return initial_lr

    def plot_lr_schedule(self):
        """Visualize learning rate changes"""
        scheduler = self.create_lr_scheduler()
        if isinstance(scheduler, (float, int)):
            return

        steps = range(self.gan_epochs.value())
        lr_values = [scheduler(step).numpy() for step in steps]

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(steps, lr_values)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        self.canvas.draw()

    def update_metrics(self, y_pred):
        """Metric update function"""
        from sklearn.metrics import f1_score

        metrics_text = "Model Performance Metrics:\n\n"

        if isinstance(self.current_model, LinearRegression):
            pass
        else:  # Classification
            accuracy = accuracy_score(self.y_test, y_pred)
            if len(np.unique(self.y_test)) == 2:
                f1 = f1_score(self.y_test, y_pred)
                metrics_text += f"F1 Score: {f1:.4f}\n"
            else:
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                metrics_text += f"Weighted F1 Score: {f1:.4f}\n"

            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(confusion_matrix(self.y_test, y_pred))

        self.metrics_text.setText(metrics_text)

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
