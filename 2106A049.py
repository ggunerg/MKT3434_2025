def create_algorithm_group(self, name, params):
    """Enhanced method to create algorithm parameter groups with more flexibility"""
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

        param_layout.addWidget(widget)
        param_widgets[param_name] = widget
        layout.addLayout(param_layout)

    # Enhanced Loss Function Selection
    loss_layout = QHBoxLayout()
    loss_label = QLabel("Loss Function:")
    loss_combo = QComboBox()

    # Determine loss functions based on problem type
    if "SVR" in name:  # Regression
        loss_combo.addItems([
            "MSE",  # Mean Squared Error
            "MAE",  # Mean Absolute Error
            "Huber"  # Huber Loss
        ])
    else:  # Classification
        loss_combo.addItems([
            "Cross-Entropy",
            "Hinge Loss"
        ])

    loss_layout.addWidget(loss_label)
    loss_layout.addWidget(loss_combo)
    layout.addLayout(loss_layout)
    param_widgets["loss_function"] = loss_combo

    # Train button
    train_btn = QPushButton(f"Train {name}")
    train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
    layout.addWidget(train_btn)

    group.setLayout(layout)
    return group


def create_classical_ml_tab(self):
    """Updated classical ML tab with enhanced SVM options"""
    widget = QWidget()
    layout = QGridLayout(widget)

    # Previous sections (Regression, Classification)
    # ...existing code for previous sections...

    # Enhanced SVM section
    svm_group = QGroupBox("Support Vector Machines")
    svm_layout = QVBoxLayout()

    # Classification SVM
    svm_classification_group = self.create_algorithm_group(
        "SVM Classification",
        {
            "C": "double",
            "kernel": ["linear", "rbf", "poly"],
            "degree": "int",  # for polynomial kernel
            "gamma": "double"  # kernel coefficient
        }
    )
    svm_layout.addWidget(svm_classification_group)

    # Support Vector Regression (SVR)
    svr_group = self.create_algorithm_group(
        "SVR",
        {
            "C": "double",
            "kernel": ["linear", "rbf", "poly"],
            "epsilon": "double",  # epsilon-tube for SVR
            "degree": "int",  # for polynomial kernel
            "gamma": "double"  # kernel coefficient
        }
    )
    svm_layout.addWidget(svr_group)

    svm_group.setLayout(svm_layout)
    layout.addWidget(svm_group, 1, 1)

    return widget


def train_model(self, model_name, param_widgets):
    """Enhanced model training method with loss function support"""
    try:
        # Collect parameters
        model_params = {}
        loss_function = None

        for param_name, widget in param_widgets.items():
            if param_name == "loss_function":
                loss_function = widget.currentText()
                continue

            if isinstance(widget, QSpinBox):
                model_params[param_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                model_params[param_name] = widget.value()
            elif isinstance(widget, QCheckBox):
                model_params[param_name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                model_params[param_name] = widget.currentText()

        # Determine model and training approach
        if model_name == "SVM Classification":
            # Handle different kernels
            kernel = model_params.get('kernel', 'rbf')
            del model_params['kernel']

            # Choose appropriate classification loss
            if loss_function == "Hinge Loss":
                # SVM with hinge loss (default for linear SVM)
                model = SVC(kernel=kernel, **model_params)
            else:
                # Soft-margin SVM with cross-entropy-like behavior
                model = SVC(kernel=kernel, probability=True, **model_params)

        elif model_name == "SVR":
            # Support Vector Regression
            kernel = model_params.get('kernel', 'rbf')
            del model_params['kernel']

            # Choose regression loss
            if loss_function == "MAE":
                model = SVR(kernel=kernel, loss='epsilon_insensitive', **model_params)
            elif loss_function == "Huber":
                # Huber-like loss approximation
                model = SVR(kernel=kernel, loss='epsilon_insensitive',
                            epsilon=model_params.get('epsilon', 0.1),
                            **{k: v for k, v in model_params.items() if k != 'epsilon'})
            else:  # Default to MSE
                model = SVR(kernel=kernel, **model_params)

        # Fit the model
        model.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = model.predict(self.X_test)

        # Store current model for metrics calculation
        self.current_model = model

        # Update visualization and metrics
        self.update_visualization(y_pred)
        self.update_metrics(y_pred)

        self.status_bar.showMessage(f"{model_name} Training Complete")

    except Exception as e:
        self.show_error(f"Error training {model_name}: {str(e)}")


# Add necessary imports at the top of the file
from sklearn.svm import SVC, SVR