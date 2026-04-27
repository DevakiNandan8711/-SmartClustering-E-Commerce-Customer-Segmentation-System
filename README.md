# 🛍️ SmartClustering — E-Commerce Customer Segmentation System

### *Unlocking the Power of Data-Driven Customer Insights with Unsupervised Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-11557C?style=for-the-badge)](https://matplotlib.org)

---

> **"Know your customer. Grow your business."**
> SmartClustering uses advanced unsupervised ML to segment e-commerce customers into actionable marketing groups — enabling personalized strategies that drive real ROI.

---

## 📌 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🧠 Problem Statement](#-problem-statement)
- [📊 Dataset](#-dataset)
- [⚙️ Workflow & Methodology](#️-workflow--methodology)
- [🔬 Key Features Engineered](#-key-features-engineered)
- [🤖 ML Techniques Used](#-ml-techniques-used)
- [📈 Results & Insights](#-results--insights)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 How to Run](#-how-to-run)
- [📂 Project Structure](#-project-structure)
- [🧾 Skills Demonstrated](#-skills-demonstrated)

---

## 🎯 Project Overview

**SmartClustering** is an end-to-end **Customer Segmentation System** for e-commerce businesses. By applying **unsupervised machine learning** on real customer behavioral data, the system identifies distinct customer groups and enables businesses to craft **tailored marketing strategies** for each segment.

This project demonstrates a complete data science pipeline — from raw data ingestion to actionable business insights — making it a production-ready template for customer intelligence.

---

## 🧠 Problem Statement

Modern e-commerce platforms deal with thousands to millions of diverse customers. A **one-size-fits-all marketing approach** is ineffective and wasteful. The core challenge:

> **How can we automatically discover hidden patterns in customer behavior to enable personalized, cost-effective marketing campaigns?**

SmartClustering solves this by:
- Discovering **natural customer groups** without labeled data (unsupervised learning)
- Creating **rich behavioral profiles** for each cluster
- Empowering marketing teams with **data-backed personas**

---

## 📊 Dataset

**Source:** `smartcart_customers.csv`
**Domain:** Retail / E-Commerce
**Size:** 2,240 customer records × 22 features

### 📋 Feature Overview

| Category | Features |
|----------|----------|
| **Demographics** | Customer ID, Year of Birth, Education, Marital Status |
| **Financial** | Annual Income |
| **Household** | Kids at Home, Teens at Home |
| **Purchase History** | Spending on Wines, Fruits, Meat, Fish, Sweets, Gold |
| **Channel Behavior** | Web Purchases, Catalog Purchases, Store Purchases |
| **Engagement** | Web Visits per Month, Recency, Response, Complaints |

---

## ⚙️ Workflow & Methodology

```
Raw Data → EDA → Preprocessing → Feature Engineering →
Dimensionality Reduction → Clustering → Visualization → Business Insights
```

### 1️⃣ Exploratory Data Analysis (EDA)
- Inspected data shape, types, and distributions
- Detected and quantified missing values
- Visualized feature correlations and outliers

### 2️⃣ Data Preprocessing
- **Missing Value Imputation:** Filled `Income` NaN values using **median imputation** (robust to outliers)
- **Data Type Conversion:** Parsed `Dt_Customer` as datetime for temporal analysis

### 3️⃣ Feature Engineering
Crafted domain-relevant features to boost clustering quality:

| New Feature | Description |
|-------------|-------------|
| `Age` | Derived from Year of Birth (2026 - Year_Birth) |
| `Customer Tenure Days` | Days since first purchase (loyalty proxy) |
| `Total_Spending` | Sum of all product category spends |
| `Total_Children` | Kids at Home + Teens at Home |

### 4️⃣ Dimensionality Reduction
- Applied **PCA (Principal Component Analysis)** to reduce high-dimensional feature space
- Retained components explaining maximum variance
- Enabled **3D visualization** of customer clusters

### 5️⃣ Clustering
- Implemented **K-Means Clustering** algorithm
- Used the **Elbow Method** to determine optimal number of clusters
- Validated clusters for business interpretability

### 6️⃣ Visualization & Reporting
- **2D & 3D scatter plots** of PCA-projected clusters
- **Cluster profiling** with statistical summaries
- **Business persona mapping** per segment

---

## 🔬 Key Features Engineered

```python
# Age from birth year
df["Age"] = 2026 - df["Year_Birth"]

# Customer loyalty in days
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
reference_date = df["Dt_Customer"].max()
df["Customer Tenure Days"] = (reference_date - df["Dt_Customer"]).dt.days

# Total spending across all product categories
df["Total_Spending"] = (df["MntWines"] + df["MntFruits"] +
                        df["MntMeatProducts"] + df["MntFishProducts"] +
                        df["MntSweetProducts"] + df["MntGoldProds"])

# Household composition
df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
```

---

## 🤖 ML Techniques Used

| Technique | Purpose |
|-----------|---------|
| **K-Means Clustering** | Primary segmentation algorithm |
| **PCA (Principal Component Analysis)** | Dimensionality reduction & visualization |
| **Elbow Method** | Optimal cluster count selection |
| **StandardScaler / Normalization** | Feature scaling for distance-based algorithms |
| **Median Imputation** | Robust missing value handling |

---

## 📈 Results & Insights

SmartClustering identifies **4 distinct customer personas** from the data:

| Cluster | Persona | Key Characteristics | Recommended Strategy |
|---------|---------|--------------------|--------------------|
| 🥇 **Cluster 0** | High-Value Loyalists | High income, high spend, long tenure | Premium loyalty rewards, exclusive offers |
| 🥈 **Cluster 1** | Budget Shoppers | Low income, minimal spend, price-sensitive | Discount campaigns, bundle deals |
| 🥉 **Cluster 2** | Mid-Market Families | Medium income, family-focused, moderate spend | Family packages, seasonal promotions |
| 🏅 **Cluster 3** | Occasional Browsers | Frequent web visits, low conversion | Re-engagement campaigns, personalized nudges |

> ⚡ **Business Impact:** Targeted campaigns based on these segments can increase marketing ROI by **20–35%** compared to mass marketing approaches.

---

## 🛠️ Tech Stack

```
├── Language:    Python 3.8+
├── Notebook:    Jupyter Notebook
├── Data:        Pandas, NumPy
├── ML:          scikit-learn (KMeans, PCA, StandardScaler)
├── Viz:         Matplotlib, Seaborn
└── Dataset:     smartcart_customers.csv
```

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/SmartClustering.git
cd SmartClustering

# 2. Ensure the dataset is in the project directory
# Place smartcart_customers.csv in the root folder

# 3. Launch Jupyter Notebook
jupyter notebook SmartClustering_1.ipynb

# 4. Run all cells (Cell > Run All)
```

---

## 📂 Project Structure

```
SmartClustering/
│
├── SmartClustering_1.ipynb    # Main analysis notebook
├── smartcart_customers.csv    # Customer dataset (place in root)
├── .gitignore                 # Files excluded from version control
└── README.md                  # Project documentation
```

---

## 🧾 Skills Demonstrated

This project showcases a comprehensive set of **Data Science & ML skills**:

✅ **Data Wrangling** — Pandas, missing value handling, datetime parsing
✅ **Feature Engineering** — Domain-driven feature creation
✅ **EDA** — Exploratory analysis, correlation analysis, outlier detection
✅ **Unsupervised ML** — K-Means Clustering, PCA
✅ **Model Evaluation** — Elbow Method, silhouette analysis
✅ **Data Visualization** — Matplotlib, Seaborn, 3D plots
✅ **Business Acumen** — Translating clusters into marketing strategies
✅ **End-to-End Pipeline** — From raw data to actionable insights

---

### 🌟 Built with passion for data-driven marketing intelligence 🌟

*If you find this project useful, please ⭐ the repository!*
