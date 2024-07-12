# titanic-survival-prediction
# Titanic Survival Prediction

This project aims to predict the survival of passengers from the Titanic dataset using logistic regression. The program includes data preprocessing, feature scaling, dimensionality reduction using PCA, model training, and evaluation.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Visualizations](#visualizations)

## Installation

1. Clone the repository to your local machine.
    ```bash
    git clone <repository_url>
    ```
2. Navigate to the project directory.
    ```bash
    cd titanic-survival-prediction
    ```
3. Install the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure that you have the Titanic dataset CSV files (`train.csv`, `test.csv`, `gender_submission.csv`) in the correct directory (`C:/Users/Asus/Desktop/data Cleaning/`).

2. Run the Python script.
    ```bash
    python titanic_survival_prediction.py
    ```

## Project Structure

titanic-survival-prediction/
├── titanic_survival_prediction.py
├── requirements.txt
├── README.md
└── data/
├── train.csv
├── test.csv
└── gender_submission.csv


## Data Preprocessing

- **Load Dataset**: The training dataset is loaded from `train.csv`.
- **Missing Values**: Missing values in the 'Age' column are filled with the mean age.
- **Categorical Data**: The 'Sex' column is converted into numerical format using one-hot encoding.

## Model Training and Evaluation

- **Feature Selection**: The selected features are 'Pclass', 'Sex_male', 'Age', and 'Fare'.
- **Feature Scaling**: The features are standardized using `StandardScaler`.
- **Dimensionality Reduction**: PCA is applied to retain 95% of the variance.
- **Model**: A logistic regression model is trained on the processed data.
- **Evaluation**: The model's performance is evaluated using accuracy, confusion matrix, and classification report.

## Visualizations

Several visualizations are created to understand the predictions and survival statistics:

- Distribution of predictions for survival status.
- Survival rate by gender.
- Survival rate by age.
- Survival rate by passenger class.
- Confusion matrix.
- Accuracy comparison between the training set and the test set.

## Output

The script will generate the following outputs:
- Console output with accuracy, confusion matrix, classification report, and predictions.
- A CSV file (`titanic_predictions.csv`) containing the predictions.
- Various visualizations displayed using matplotlib and seaborn.

## Example Results

```plaintext
Accuracy:  0.8212290502793296
Confusion Matrix:
 [[105  14]
 [ 24  56]]
Classification Report:
               precision    recall  f1-score   support

           0       0.81      0.88      0.85       119
           1       0.80      0.70      0.75        80

    accuracy                           0.81       199
   macro avg       0.81      0.79      0.80       199
weighted avg       0.81      0.81      0.81       199



Feel free to adjust the content to better fit your specific needs and project details.
