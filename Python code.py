# BHARATPAY: PREDICTIVE ANALYTICS OF DIGITAL PAYMENT TRANSACTIONS

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="deep")

RANDOM_STATE = 42
TEST_SIZE = 0.20

file_path = (
    r"D:\B.Tech\Sem 5\Predictive Analysis\CA-2\npci-daily.xlsx"
)

if not os.path.isfile(file_path):
    raise FileNotFoundError(
        "Dataset not found. Check the file path."
    )

output_folder = os.path.dirname(file_path)

target = "UPI, number of transactions, daily"

payment_columns = [
    "UPI, number of transactions, daily",
    "IMPS, number of transactions, daily",
    "FASTag (NETC) toll and parking, number of transactions, daily",
    "ATM network (NFS) transactions, number of transactions, daily",
    "AePS cash withdrawals, number of transactions, daily"
]


# STEP 2: LOAD AND INSPECT DATA

print("\n" + "=" * 70)
print("BHARATPAY PREDICTIVE ANALYTICS")
print("=" * 70)

df = pd.read_excel(file_path)

print("\nOriginal Dataset Shape:", df.shape)

if "date" not in df.columns:
    raise ValueError("Date column not found.")

if target not in df.columns:
    raise ValueError("UPI target column not found.")

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

invalid_dates = df["date"].isna().sum()

df = df.dropna(subset=["date"]).copy()
df = df.sort_values("date").reset_index(drop=True)

duplicate_count = df.duplicated().sum()

df = df.drop_duplicates().reset_index(drop=True)

# Convert columns to numeric
for column in df.columns:
    if column != "date":
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

# Missing-value handling
missing_before = df[numeric_columns].isna().sum().sum()

df[numeric_columns] = df[numeric_columns].interpolate(
    method="linear",
    limit_direction="both"
)

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

missing_after = df[numeric_columns].isna().sum().sum()

df = df.dropna(subset=[target])
df = df[df[target] >= 0].reset_index(drop=True)

# Verify payment columns
missing_columns = [
    col for col in payment_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing payment columns: {missing_columns}"
    )

print("\nDATA QUALITY REPORT")
print("-" * 70)

print("Records:", len(df))
print("Columns:", df.shape[1])
print("Invalid Dates:", invalid_dates)
print("Duplicate Rows:", duplicate_count)
print("Missing Values Before:", missing_before)
print("Missing Values After:", missing_after)
print("Start Date:", df["date"].min())
print("End Date:", df["date"].max())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe().T)

df.to_excel(
    os.path.join(
        output_folder,
        "BharatPay_Preprocessed.xlsx"
    ),
    index=False
)


# STEP 3: EXPLORATORY DATA ANALYSIS

print("\nEXPLORATORY DATA ANALYSIS")

# 3.1 UPI transaction trend

plt.figure(figsize=(14, 5))

plt.plot(
    df["date"],
    df[target],
    linewidth=1.5
)

plt.title("Daily UPI Transaction Volume in India")
plt.xlabel("Date")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "01_UPI_Transaction_Trend.png"
    ),
    dpi=300
)

plt.show()

# 3.2 UPI transaction value

upi_value_column = "UPI, value, daily"

if upi_value_column in df.columns:

    plt.figure(figsize=(14, 5))

    plt.plot(
        df["date"],
        df[upi_value_column],
        linewidth=1.5
    )

    plt.title("Daily UPI Transaction Value")
    plt.xlabel("Date")
    plt.ylabel("Transaction Value")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "02_UPI_Transaction_Value.png"
        ),
        dpi=300
    )

    plt.show()

# 3.3 Payment method comparison

comparison_columns = {
    "UPI": "UPI, number of transactions, daily",
    "IMPS": "IMPS, number of transactions, daily",
    "FASTag": "FASTag (NETC) toll and parking, number of transactions, daily",
    "ATM": "ATM network (NFS) transactions, number of transactions, daily",
    "AePS": "AePS cash withdrawals, number of transactions, daily"
}

payment_data = pd.DataFrame({
    "Payment Method": list(comparison_columns.keys()),
    "Average Daily Transactions": [
        df[column].mean()
        for column in comparison_columns.values()
    ]
})

payment_data = payment_data.sort_values(
    "Average Daily Transactions",
    ascending=False
)

plt.figure(figsize=(10, 5))

sns.barplot(
    data=payment_data,
    x="Payment Method",
    y="Average Daily Transactions"
)

plt.title("Average Daily Transactions by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Average Transactions")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "03_Payment_Method_Comparison.png"
    ),
    dpi=300
)

plt.show()

# 3.4 Correlation heatmap

correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.3
)

plt.title("Correlation Between Digital Payment Indicators")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "04_Correlation_Heatmap.png"
    ),
    dpi=300
)

plt.show()

correlation_matrix.to_excel(
    os.path.join(
        output_folder,
        "BharatPay_Correlation_Matrix.xlsx"
    )
)


# STEP 4: FEATURE ENGINEERING

print("\nFEATURE ENGINEERING")

model_data = df.copy()

# Calendar features
model_data["year"] = model_data["date"].dt.year
model_data["month"] = model_data["date"].dt.month
model_data["day_of_week"] = model_data["date"].dt.dayofweek
model_data["day_of_year"] = model_data["date"].dt.dayofyear

model_data["is_weekend"] = (
    model_data["date"].dt.dayofweek >= 5
).astype(int)

feature_columns = [
    "year",
    "month",
    "day_of_week",
    "day_of_year",
    "is_weekend"
]

# Historical lag features
for column in numeric_columns:

    for lag in [1, 7, 14]:

        new_column = f"{column}_lag{lag}"

        model_data[new_column] = (
            model_data[column].shift(lag)
        )

        feature_columns.append(new_column)

# Rolling averages using past values only
model_data["UPI_rolling_mean_7"] = (
    model_data[target]
    .shift(1)
    .rolling(window=7)
    .mean()
)

model_data["UPI_rolling_mean_14"] = (
    model_data[target]
    .shift(1)
    .rolling(window=14)
    .mean()
)

model_data["UPI_rolling_std_7"] = (
    model_data[target]
    .shift(1)
    .rolling(window=7)
    .std()
)

feature_columns += [
    "UPI_rolling_mean_7",
    "UPI_rolling_mean_14",
    "UPI_rolling_std_7"
]

# Predict daily change rather than the complete transaction count
model_data["UPI_Daily_Change"] = (
    model_data[target].diff()
)

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna(
    subset=feature_columns + [
        "UPI_Daily_Change",
        target
    ]
).reset_index(drop=True)

X = model_data[feature_columns].copy()

y_change = model_data["UPI_Daily_Change"].copy()

y_actual = model_data[target].copy()

previous_day_values = model_data[
    target + "_lag1"
].copy()

print("Feature Engineering Completed!")
print("Model Dataset Shape:", model_data.shape)
print("Number of Features:", len(feature_columns))
print("Target Variable:", target)
print("Prediction Target: Daily UPI Transaction Change")

model_data.to_excel(
    os.path.join(
        output_folder,
        "BharatPay_Feature_Engineered.xlsx"
    ),
    index=False
)


# STEP 5: CHRONOLOGICAL TRAIN-TEST SPLIT

print("\nTRAIN-TEST SPLIT")

split_index = int(
    len(model_data) * (1 - TEST_SIZE)
)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y_change.iloc[:split_index].copy()

y_test_change = y_change.iloc[split_index:].copy()

y_test_actual = y_actual.iloc[split_index:].copy()

previous_day_test = previous_day_values.iloc[
    split_index:
].copy()

test_dates = model_data["date"].iloc[
    split_index:
].copy()

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

print(
    "Training Period:",
    model_data["date"].iloc[0],
    "to",
    model_data["date"].iloc[split_index - 1]
)

print(
    "Testing Period:",
    model_data["date"].iloc[split_index],
    "to",
    model_data["date"].iloc[-1]
)


# STEP 6: BASELINE MODEL

print("\nBASELINE MODEL")

# Today's count is assumed equal to yesterday's count
baseline_predictions = previous_day_test.to_numpy()

baseline_mae = mean_absolute_error(
    y_test_actual,
    baseline_predictions
)

baseline_mse = mean_squared_error(
    y_test_actual,
    baseline_predictions
)

baseline_rmse = np.sqrt(baseline_mse)

baseline_r2 = r2_score(
    y_test_actual,
    baseline_predictions
)

print("Previous-Day Baseline Results")

print("MAE:", baseline_mae)
print("MSE:", baseline_mse)
print("RMSE:", baseline_rmse)
print("R2 Score:", baseline_r2)


# STEP 7: MACHINE LEARNING MODEL TRAINING

print("\nMACHINE LEARNING MODEL TRAINING")

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        loss="huber",
        random_state=RANDOM_STATE
    ),

    "Support Vector Regression": make_pipeline(
        StandardScaler(),
        SVR(
            kernel="rbf",
            C=10,
            epsilon=0.05,
            gamma="scale"
        )
    )
}

results = []

predictions_dictionary = {}

trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    try:

        model.fit(
            X_train,
            y_train
        )

        predicted_change = model.predict(X_test)

        # Reconstruct original transaction count
        predictions = (
            previous_day_test.to_numpy()
            + predicted_change
        )

        predictions = np.maximum(
            predictions,
            0
        )

        mae = mean_absolute_error(
            y_test_actual,
            predictions
        )

        mse = mean_squared_error(
            y_test_actual,
            predictions
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test_actual,
            predictions
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        })

        predictions_dictionary[name] = predictions

        trained_models[name] = model

        print("MAE:", mae)
        print("RMSE:", rmse)
        print("R2 Score:", r2)

    except Exception as error:

        print("Model failed:", name)
        print("Reason:", error)

if not results:
    raise RuntimeError(
        "No regression model completed successfully."
    )

results_df = pd.DataFrame(results)

baseline_row = pd.DataFrame([{
    "Model": "Previous-Day Baseline",
    "MAE": baseline_mae,
    "MSE": baseline_mse,
    "RMSE": baseline_rmse,
    "R2 Score": baseline_r2
}])

comparison_df = pd.concat(
    [baseline_row, results_df],
    ignore_index=True
)

comparison_df = comparison_df.sort_values(
    "RMSE"
).reset_index(drop=True)

print("\nMODEL PERFORMANCE COMPARISON")

print(
    comparison_df.to_string(index=False)
)

comparison_df.to_excel(
    os.path.join(
        output_folder,
        "BharatPay_Model_Comparison.xlsx"
    ),
    index=False
)

# Select best ML model based on RMSE
best_model_name = min(
    predictions_dictionary,
    key=lambda name: mean_squared_error(
        y_test_actual,
        predictions_dictionary[name]
    )
)

best_model = trained_models[best_model_name]

best_predictions = predictions_dictionary[
    best_model_name
]

best_residuals = (
    y_test_actual.to_numpy()
    - best_predictions
)

best_r2 = r2_score(
    y_test_actual,
    best_predictions
)

best_mae = mean_absolute_error(
    y_test_actual,
    best_predictions
)

best_rmse = np.sqrt(
    mean_squared_error(
        y_test_actual,
        best_predictions
    )
)

print("\nBEST MACHINE LEARNING MODEL:", best_model_name)

print("Best Model MAE:", best_mae)
print("Best Model RMSE:", best_rmse)
print("Best Model R2:", best_r2)


# STEP 8: BEST MODEL EVALUATION

print("\nBEST MODEL EVALUATION")

print(
    "Mean Residual:",
    np.mean(best_residuals)
)

print(
    "Maximum Absolute Error:",
    np.max(np.abs(best_residuals))
)

# Actual vs predicted trend
plt.figure(figsize=(14, 6))

plt.plot(
    test_dates,
    y_test_actual.to_numpy(),
    label="Actual UPI Transactions",
    linewidth=1.7
)

plt.plot(
    test_dates,
    best_predictions,
    label="Predicted UPI Transactions",
    linewidth=1.7
)

plt.title(
    f"{best_model_name}: Actual vs Predicted UPI Transactions"
)

plt.xlabel("Date")
plt.ylabel("Transaction Count")

plt.legend()

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "05_Actual_vs_Predicted.png"
    ),
    dpi=300
)

plt.show()

# Residual distribution
plt.figure(figsize=(10, 5))

sns.histplot(
    best_residuals,
    kde=True
)

plt.title("Residual Error Distribution")
plt.xlabel("Actual - Predicted Transactions")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "06_Residual_Distribution.png"
    ),
    dpi=300
)

plt.show()

# Actual vs predicted scatter plot
plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=y_test_actual.to_numpy(),
    y=best_predictions,
    alpha=0.6
)

min_value = min(
    float(y_test_actual.min()),
    float(best_predictions.min())
)

max_value = max(
    float(y_test_actual.max()),
    float(best_predictions.max())
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title("Actual vs Predicted UPI Transactions")

plt.xlabel("Actual Transactions")
plt.ylabel("Predicted Transactions")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "07_Actual_Predicted_Scatter.png"
    ),
    dpi=300
)

plt.show()


# STEP 9: MODEL COMPARISON VISUALIZATIONS

ml_comparison = comparison_df[
    comparison_df["Model"] != "Previous-Day Baseline"
].copy()

# R2 comparison
plt.figure(figsize=(12, 6))

sns.barplot(
    data=ml_comparison,
    x="Model",
    y="R2 Score"
)

plt.title("R2 Score Comparison of Machine Learning Models")

plt.xlabel("Machine Learning Model")
plt.ylabel("R2 Score")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "08_R2_Comparison.png"
    ),
    dpi=300
)

plt.show()

# RMSE comparison
plt.figure(figsize=(12, 6))

sns.barplot(
    data=ml_comparison,
    x="Model",
    y="RMSE"
)

plt.title("RMSE Comparison of Machine Learning Models")

plt.xlabel("Machine Learning Model")
plt.ylabel("Root Mean Squared Error")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "09_RMSE_Comparison.png"
    ),
    dpi=300
)

plt.show()

# MAE comparison
plt.figure(figsize=(12, 6))

sns.barplot(
    data=ml_comparison,
    x="Model",
    y="MAE"
)

plt.title("MAE Comparison of Machine Learning Models")

plt.xlabel("Machine Learning Model")
plt.ylabel("Mean Absolute Error")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "10_MAE_Comparison.png"
    ),
    dpi=300
)

plt.show()

# Save predictions
prediction_results = pd.DataFrame({

    "Date": test_dates.to_numpy(),

    "Actual UPI Transactions": y_test_actual.to_numpy(),

    "Previous-Day Baseline": baseline_predictions,

    "Best Model": best_model_name,

    "Best Model Prediction": best_predictions,

    "Residual": best_residuals

})

for model_name, prediction_values in predictions_dictionary.items():

    prediction_results[
        model_name + " Prediction"
    ] = prediction_values

prediction_results.to_excel(
    os.path.join(
        output_folder,
        "BharatPay_Prediction_Results.xlsx"
    ),
    index=False
)

print("\nModel comparison charts generated.")
print("Prediction results saved.")


# STEP 10: K-MEANS CLUSTERING

print("\nK-MEANS CLUSTERING")

cluster_data = df[payment_columns].copy()

cluster_data = cluster_data.replace(
    [np.inf, -np.inf],
    np.nan
)

cluster_data = cluster_data.fillna(
    cluster_data.median()
)

# Log transformation
cluster_log_data = np.log1p(
    cluster_data.clip(lower=0)
)

cluster_scaler = StandardScaler()

scaled_cluster_data = cluster_scaler.fit_transform(
    cluster_log_data
)

cluster_scores = []

for k in range(2, 7):

    kmeans_test = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20
    )

    labels = kmeans_test.fit_predict(
        scaled_cluster_data
    )

    score = silhouette_score(
        scaled_cluster_data,
        labels,
        sample_size=min(
            2000,
            len(scaled_cluster_data)
        ),
        random_state=RANDOM_STATE
    )

    cluster_sizes = pd.Series(labels).value_counts()

    cluster_scores.append({

        "K": k,

        "Inertia": kmeans_test.inertia_,

        "Silhouette Score": score,

        "Smallest Cluster Size": int(
            cluster_sizes.min()
        )

    })

    print(
        f"K={k} | "
        f"Inertia={kmeans_test.inertia_:.2f} | "
        f"Silhouette={score:.4f} | "
        f"Smallest Cluster={cluster_sizes.min()}"
    )

cluster_scores_df = pd.DataFrame(
    cluster_scores
)

optimal_k = int(
    cluster_scores_df.loc[
        cluster_scores_df[
            "Silhouette Score"
        ].idxmax(),
        "K"
    ]
)

print("\nSelected Number of Clusters:", optimal_k)

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=RANDOM_STATE,
    n_init=20
)

cluster_labels = kmeans.fit_predict(
    scaled_cluster_data
)

clustered_df = df.copy()

clustered_df["Payment_Cluster"] = cluster_labels

cluster_counts = (
    clustered_df["Payment_Cluster"]
    .value_counts()
    .sort_index()
)

print("\nCLUSTER DISTRIBUTION")

print(cluster_counts)

cluster_scores_df.to_excel(
    os.path.join(
        output_folder,
        "BharatPay_Optimal_Clusters.xlsx"
    ),
    index=False
)

# Cluster distribution graph
plt.figure(figsize=(9, 5))

sns.barplot(
    x=cluster_counts.index.astype(str),
    y=cluster_counts.values
)

plt.title("Distribution of Daily Payment Activity Clusters")

plt.xlabel("Cluster")
plt.ylabel("Number of Days")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "12_Cluster_Distribution.png"
    ),
    dpi=300
)

plt.show()


# STEP 11: PRINCIPAL COMPONENT ANALYSIS

print("\nPRINCIPAL COMPONENT ANALYSIS")

pca = PCA(
    n_components=2
)

pca_data = pca.fit_transform(
    scaled_cluster_data
)

explained_variance = pca.explained_variance_ratio_

total_variance = explained_variance.sum()

print(
    "Explained Variance Ratio:",
    explained_variance
)

print(
    "Total Variance Explained:",
    total_variance
)

pca_plot_df = pd.DataFrame({

    "Principal Component 1": pca_data[:, 0],

    "Principal Component 2": pca_data[:, 1],

    "Payment Cluster": cluster_labels.astype(str)

})

plt.figure(figsize=(11, 7))

sns.scatterplot(
    data=pca_plot_df,
    x="Principal Component 1",
    y="Principal Component 2",
    hue="Payment Cluster",
    palette="viridis",
    s=40,
    alpha=0.75
)

plt.title(
    "Digital Payment Pattern Clustering Using K-Means and PCA"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.legend(title="Payment Cluster")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "11_KMeans_PCA_Clusters.png"
    ),
    dpi=300
)

plt.show()

pca_plot_df.to_excel(
    os.path.join(
        output_folder,
        "BharatPay_PCA_Results.xlsx"
    ),
    index=False
)


# STEP 12: CLUSTER PROFILING

print("\nCLUSTER PROFILING")

cluster_summary = clustered_df.groupby(
    "Payment_Cluster"
)[payment_columns].mean()

cluster_summary["Number of Days"] = cluster_counts

print("\nCLUSTER PROFILE")

print(cluster_summary)

cluster_summary.to_excel(
    os.path.join(
        output_folder,
        "BharatPay_Cluster_Analysis.xlsx"
    )
)

clustered_df.to_excel(
    os.path.join(
        output_folder,
        "BharatPay_Dataset_With_Clusters.xlsx"
    ),
    index=False
)


# STEP 13: SMALLEST CLUSTER INVESTIGATION

print("\nSMALLEST CLUSTER INVESTIGATION")

smallest_cluster = int(
    cluster_counts.idxmin()
)

smallest_cluster_data = clustered_df[
    clustered_df["Payment_Cluster"] == smallest_cluster
].copy()

print("Smallest Cluster:", smallest_cluster)

print(
    "Number of Records:",
    len(smallest_cluster_data)
)

print("\nSMALLEST CLUSTER DETAILS")

print(
    smallest_cluster_data[
        ["date"] + payment_columns
    ].to_string(index=False)
)

cluster_comparison = pd.DataFrame({

    "Smallest Cluster Mean":
        smallest_cluster_data[payment_columns].mean(),

    "Dataset Median":
        df[payment_columns].median(),

    "Dataset Maximum":
        df[payment_columns].max()

})

print("\nCLUSTER COMPARISON")

print(cluster_comparison)

cluster_comparison.to_excel(
    os.path.join(
        output_folder,
        "BharatPay_Smallest_Cluster_Investigation.xlsx"
    )
)


# STEP 14: FINAL PROJECT SUMMARY

print("\n" + "=" * 70)
print("BHARATPAY PROJECT EXECUTION SUMMARY")
print("=" * 70)
print("Dataset Records:", len(df))
print("Dataset Features:", df.shape[1])
print("Engineered Model Records:", len(model_data))
print("Model Features:", len(feature_columns))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Regression Models Trained:", len(trained_models))
print("Best Model:", best_model_name)
print("Best Model MAE:", best_mae)
print("Best Model RMSE:", best_rmse)
print("Best Model R2:", best_r2)
print("Baseline R2:", baseline_r2)
print("Optimal K-Means Clusters:", optimal_k)
print("PCA Variance Explained:", total_variance)
print("\nAll outputs saved in:")
print(output_folder)
print("\nBHARATPAY PREDICTIVE ANALYTICS COMPLETED SUCCESSFULLY!")
print("=" * 70)