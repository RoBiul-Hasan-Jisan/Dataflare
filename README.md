# DataFlare ML Studio

A production-ready AutoML platform built with Flask that enables users to upload datasets, perform automated machine learning, and export trained models without writing a single line of code.

![DataForge ML Studio Screenshot](https://screenshot.png)

---

##  Features

###  Data Exploration
* **Upload & Preview:** Support for CSV and Excel files up to 100MB.
* **Smart Data Profiling:** Automatic detection of data types, missing values, and duplicates.
* **Quick Cleaning:** One-click operations to drop duplicates and empty columns.
* **Statistical Summary:** Comprehensive statistics for numerical columns.

###  Exploratory Data Analysis (EDA)
* **Distribution Plots:** Histograms, box plots, and violin plots for numerical data.
* **Categorical Analysis:** Bar charts for categorical columns.
* **Correlation Heatmap:** Interactive correlation matrix for numerical features.
* **Column Statistics:** Detailed statistics including mean, median, quartiles, and skewness.

###  Automated Machine Learning
* **Smart Problem Detection:** Automatically detects classification vs. regression tasks.
* **15+ Algorithms:** Includes XGBoost, LightGBM, CatBoost, Random Forest, and more.
* **K-Fold Cross-Validation:** Configurable fold count for robust model evaluation.
* **Memory-Safe Training:** Automatic sampling for large datasets to prevent crashes.
* **Feature Engineering:** Optional normalization and outlier removal.

###  Model Results
* **Leaderboard:** Ranked comparison of all trained models.
* **Performance Metrics:** Comprehensive metrics for each model.
* **Best Model Selection:** Automatically identifies the top-performing model.
* **Visual Analytics:** Bar charts and radar plots for model comparison.
* **Model Export:** Download trained models as pickle (`.pkl`) files.

###  Training History
* **Session History:** Tracks all training runs in the current session.
* **Export History:** Download training history as a CSV.

---

##  Technology Stack
* **Backend:** Flask (Python 3.9+)
* **ML Framework:** PyCaret 3.2.0
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly
* **Session Management:** Flask-Session
* **ML Algorithms:** XGBoost, LightGBM, CatBoost, Scikit-learn
* **Production Server:** Gunicorn
* **Containerization:** Docker

---

##  Prerequisites
* Python 3.9 or higher
* `pip` (Python package manager)
* Docker (optional, for containerized deployment)
* 4GB+ RAM recommended for training

---
## Memory Settings

The application includes memory-safe training with these defaults:

- MAX_ROWS_TRAINING = 5,000: Maximum rows for training (auto-sampled).

- MAX_ROWS_WARNING = 2,000: Warning threshold for large datasets.
---
## Usage Guide

- **Upload Data:** Click on the upload area or drag and drop a CSV/Excel file. Or select a sample dataset (Titanic, Diamonds, Iris).

- **Explore Data:** View dataset statistics, check column details, and use quick actions to clean data.

- **Perform EDA:** Select columns for distribution analysis, view correlation heatmaps, and analyze statistical summaries.

- **Train Model:** Select a target column (auto-detects problem type), configure training parameters (split ratio, CV folds, normalization, outlier removal), and click "Start Training".

- **Review Results:** View the ranked leaderboard, check performance metrics, download results as CSV, and export the trained model as a .pkl file.

- **Track History:** View all training runs in the current session, compare performance, and export history for documentation.

---

## Performance Optimization

- **Large Datasets:** Automatically sampled to 5,000 rows for training to ensure stability.

- **Memory Management:** Automatic garbage collection runs after training.

- **Concurrent Users:** Session-based isolation prevents user interference.

- **Docker Limits:** Configure memory and CPU limits in your docker-compose.yml.
---

## Troubleshooting

**Common Issues**

- Upload fails with "Unknown error": Check file size (max 100MB), verify file format (CSV or Excel), and check server logs.

- Training crashes with memory error: Reduce dataset size, decrease CV folds (2-3 folds), or enable auto-sampling.

- Correlation heatmap not showing: Ensure at least 2 numeric columns exist, check for NaN values, and verify data types.

## 👨‍💻 Author
**Robiul Hasan Jisan**

- **Portfolio:** [robiulhasanjisan.vercel.app](https://robiulhasanjisan.vercel.app/)
- **GitHub:** [@RoBiul-Hasan-Jisan](https://github.com/RoBiul-Hasan-Jisan)
- **Kaggle:** [robiulhasanjisan](https://www.kaggle.com/robiulhasanjisan)

---

## Acknowledgments

- PyCaret for the amazing AutoML library.

- Plotly for interactive visualizations.

- Flask for the robust web framework.



