# Bank Customer Churn Prediction

This project explores machine-learning approaches for predicting whether a bank customer will churn. The analysis is implemented in a Jupyter notebook and focuses on improving F1 score for this imbalanced classification problem.

## Project structure

```text
bank_churn/
├── configs.yaml          # Model configuration
├── notebook/
│   └── train.ipynb       # Exploratory analysis, training, tuning, and evaluation
└── src/
    └── preprocess/       # Preprocessing code
```

## Data preparation

The `Exited` column is the target variable. The identifier and name columns (`RowNumber`, `CustomerId`, and `Surname`) are removed because they do not provide useful predictive information.

The remaining features are processed within a scikit-learn `Pipeline`:

- Numeric columns have missing values replaced with the median and are standardized with `StandardScaler`.
- Categorical columns have missing values replaced with `"missing"` and are converted to numeric indicator columns with `OneHotEncoder`.
- `handle_unknown="ignore"` allows the validation and test sets to contain categories that were not present during training.

Using the preprocessor inside the pipeline ensures that transformations are fitted only on the training data, preventing data leakage.

## Models

The notebook compares several scikit-learn classifiers:

- Logistic Regression
- Random Forest
- Gradient Boosting

Because churn data is imbalanced, the notebook evaluates class-weighted models and minority-class upsampling. Hyperparameters are tuned with five-fold cross-validation on the training set, using F1 score as the primary model-selection metric. A separate validation set is used to compare the tuned models.

## Results

The class-weighted Random Forest produced the best validation result, with an F1 score of approximately **0.59**. It was selected for final evaluation and achieved an F1 score of approximately **0.50** on the held-out test set.

The test set should be used only for final evaluation. It must not be used for hyperparameter tuning, choosing the sampling strategy, or selecting a classification threshold.

## Getting started

1. Install Python and create an environment.
2. Install the required packages:

   ```bash
   pip install jupyter pandas numpy scikit-learn matplotlib seaborn pyyaml tensorflow
   ```

3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open [`notebook/train.ipynb`](notebook/train.ipynb) and run the cells in order.

## Configuration

Model defaults are stored in [`configs.yaml`](configs.yaml). The configuration includes parameters for the Random Forest, Logistic Regression, and Gradient Boosting models, including random seeds and tree settings.

## Evaluation

The notebook reports precision, recall, and F1 score. F1 is calculated as the harmonic mean of precision and recall:

```text
F1 = 2 × (precision × recall) / (precision + recall)
```

This metric is useful here because it balances identifying churners with avoiding excessive false-positive predictions.