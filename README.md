# Bank Customer Churn Prediction

This project builds and compares machine-learning models to predict whether a bank customer will churn (`Exited = 1`). The workflow includes exploratory analysis, preprocessing, model tuning, and evaluation with emphasis on the F1 score for this imbalanced classification task. While working on the project I learned SHAP, Pipeline function to tuning different models efficiently.

## Project structure

```text
bank_churn/
├── README.md
├── configs.yaml                     # Model configuration and hyperparameter defaults
├── requirements.txt                 # Python dependencies
├── hypertuning.ipynb                # Hyperparameter tuning and model comparison
├── notebook/
│   └── train.ipynb                 # Earlier notebook version of the workflow
├── src/
│   ├── bank_train.ipynb             # Final model training and evaluation notebook
│   └── train.ipynb                  # Additional training notebook
├── test/
│   └── ...                          # Test-related files and outputs
└── .vscode/
   └── settings.json
```

## Objective

The goal is to predict customer churn using customer profile and account features such as age, balance, tenure, geography, and product usage. The target variable is `Exited`.

## Data preparation

The raw dataset is cleaned before modeling:

- Identifier and non-informative columns such as `RowNumber`, `CustomerId`, and `Surname` are removed.
- Numerical features are processed with imputation and scaling.
- Categorical features are processed with imputation and one-hot encoding.
- The model pipeline is fit on the training data only to avoid data leakage.

The train/validation/test split is performed before hyperparameter tuning and evaluation.

## Models evaluated

The notebooks compare several scikit-learn classifiers:

- Logistic Regression
- Random Forest
- Gradient Boosting

Because churn is an imbalanced classification problem, the project also evaluates:

- class-weighted models
- minority-class upsampling

F1 score is used as the primary model-selection metric.

## Results

The best-performing validation configuration in the final project workflow was the class-weighted Random Forest, which reached approximately:

- F1 score: **0.61**
- Accuracy: **0.84**
- Precision: **0.62**
- Recall: **0.61**
- ROC AUC: **0.86**

On the held-out test set, the final F1 score was approximately **0.50**, which indicates that the model still has room for improvement before deployment in a production setting.

## Getting started

1. Create a Python environment and activate it.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook or JupyterLab:

   ```bash
   jupyter notebook
   ```

4. Open one of the notebooks, starting with `src/bank_train.ipynb` or `hypertuning.ipynb`.

## Configuration

Model settings and defaults are stored in [`configs.yaml`](configs.yaml). These include parameters for Random Forest, Logistic Regression, and Gradient Boosting models.

## Evaluation notes

The project reports several metrics, including precision, recall, F1 score, and ROC AUC. F1 is emphasized because it provides a better summary than accuracy for imbalanced datasets, where a model can appear highly accurate while missing the minority churn class.

The final evaluation should use a separate unseen test set, and the test set should not be used for hyperparameter tuning or model selection.