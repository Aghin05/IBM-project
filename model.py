import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------------------
# BASE DIRECTORY FOR SAVING FILES
# ------------------------------------------
BASE_DIR = r"D:\PROGRAMMING\IBM PROJECT\IBM PROJECT"

# ------------------------------------------
# Part 1: Load Dataset Instead of Generating
# ------------------------------------------

def load_dataset():
    file_path = os.path.join(BASE_DIR, "Salary_Dataset.csv")
    df = pd.read_csv(file_path)

    df.rename(columns={
        "JobTitle": "Job Title",
        "Education": "Education Level",
        "YearsOfExperience": "Years of Experience"
    }, inplace=True)

    # FIX: Drop rows where Salary is missing
    df.dropna(subset=['Salary'], inplace=True)

    # FIX: Also handle missing values in other columns
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Years of Experience'] = df['Years of Experience'].fillna(df['Years of Experience'].median())
    df['Gender'] = df['Gender'].fillna('Unknown')
    df['Job Title'] = df['Job Title'].fillna('Unknown')
    df['Education Level'] = df['Education Level'].fillna('Unknown')

    print("\nDataset Loaded Successfully!\n")
    print(df.head())
    return df


# ------------------------------------------
# Part 2: Data Visualization
# ------------------------------------------

def visualize_data(data):
    print("\n--- Generating Data Visualizations ---")

    output_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style("whitegrid")

    # 1. Salary Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Salary'], kde=True, bins=30)
    plt.title('Distribution of Employee Salaries', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'salary_distribution.png'))
    plt.show()

    # 2. Salary vs Experience
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Years of Experience', y='Salary', data=data, alpha=0.6)
    plt.title('Salary vs. Years of Experience', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'salary_vs_experience.png'))
    plt.show()

    # 3. Salary by Job Title
    plt.figure(figsize=(12, 8))
    sns.boxplot(y='Job Title', x='Salary', data=data)
    plt.title('Salary by Job Title', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'salary_by_job_title.png'))
    plt.show()

    print("Visualizations saved successfully!")

# ------------------------------------------
# Part 3: Model Training + Optimization
# ------------------------------------------

def train_and_save_model(data):
    print("\n--- Starting Model Training & Optimization ---")

    X = data.drop('Salary', axis=1)
    y = data['Salary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = ['Job Title', 'Education Level', 'Gender']
    numerical_features = ['Years of Experience', 'Age']

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)


    param_grid_rf = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None]
    }

    param_grid_gb = {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.05, 0.1]
    }

    pipeline_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    pipeline_gb = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])

    print("Tuning Random Forest...")
    grid_search_rf = GridSearchCV(
        pipeline_rf, param_grid_rf, cv=5, n_jobs=1, scoring='r2'
    )
    grid_search_rf.fit(X_train, y_train)

    print("Tuning Gradient Boosting...")
    grid_search_gb = GridSearchCV(
        pipeline_gb, param_grid_gb, cv=5, n_jobs=1, scoring='r2'
    )
    grid_search_gb.fit(X_train, y_train)

    best_rf = grid_search_rf.best_estimator_.named_steps['regressor']
    best_gb = grid_search_gb.best_estimator_.named_steps['regressor']

    ensemble_model = VotingRegressor(
        estimators=[
            ('gb', best_gb),
            ('rf', best_rf),
            ('lr', LinearRegression())
        ]
    )

    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ensemble_model)
    ])

    final_pipeline.fit(X_train, y_train)

    y_pred = final_pipeline.predict(X_test)

    print("\n--- Final Model Evaluation ---")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:,.2f}")


    # Save model to IBM PROJECT directory
    model_output_path = os.path.join(BASE_DIR, 'salary_prediction_model_enhanced.pkl')
    joblib.dump(final_pipeline, model_output_path)

    print(f"\nModel saved successfully at: {model_output_path}")

# ------------------------------------------
# Main Execution
# ------------------------------------------

if __name__ == "__main__":
    df = load_dataset()
    visualize_data(df)
    train_and_save_model(df)
