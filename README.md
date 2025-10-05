# 📈 Bank Telemarketing Campaign Analysis

## 🏦 Project Overview
This project analyzes data from a **Portuguese bank’s telemarketing campaign** to understand the factors influencing customers’ decisions to **subscribe to a term deposit**.  
The analysis was performed using **Python** and covers **data cleaning, exploratory analysis, feature engineering, and statistical insights**.

The dataset includes details about customer demographics, contact details, and campaign outcomes, allowing data-driven insights to improve future marketing effectiveness.

---

## 🎯 Business Objective
The objective of this analysis is to:
- Identify **key factors** influencing customer response to term deposit offers.
- Understand the **impact of demographic and behavioral factors** (age, balance, loan status, etc.) on campaign success.
- Develop a **data-driven framework** to guide future telemarketing strategies.

---

## 🧰 Tools & Libraries Used
| Category | Libraries |
|-----------|------------|
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Statistics | `scipy.stats`, `statsmodels` |
| Machine Learning (Prep) | `sklearn` |
| Time & Regex | `datetime`, `re` |

---

## 📂 Dataset Information
**Dataset Name:** `bank_marketing_updated_v1.csv`

**Key Columns:**
| Feature | Description |
|----------|-------------|
| `age` | Age of the customer |
| `jobedu` | Combination of job type and education |
| `marital` | Marital status |
| `balance` | Customer account balance |
| `loan` | Indicates if customer has a personal loan |
| `default` | Credit default history |
| `pdays` | Days since last contact |
| `previous` | Number of previous campaigns contacted |
| `response` | Target variable – whether customer subscribed (‘yes’/‘no’) |

---

## 🔍 Project Workflow

### **1. Data Loading and Cleaning**
- Loaded the dataset and adjusted column headers.
- Converted data types and handled missing or inconsistent values.
- Combined day and month columns to create a unified `date` field.
- Converted `duration` into seconds for uniformity.
- Removed duplicate rows and irrelevant data.

### **2. Descriptive Statistics**
- Generated key metrics (mean, median, std deviation) for numerical variables.
- Evaluated the distribution of `response` (target variable).

### **3. Univariate Analysis**
Explored distribution and spread of major numerical features:
- Plotted **Boxplots**, **Histograms**, and **Density (KDE)** plots for:
  - `age`
  - `balance`
  - `duration_sec`

### **4. Bivariate Analysis**
- Analyzed correlation of numeric variables with the target.
- Explored relationships between:
  - **Numerical vs Target** → scatter and correlation plots.
  - **Categorical vs Target** → bar plots, box plots, and mean response charts.
- Used **pair plots** to visualize variable relationships.

### **5. Categorical Variable Analysis**
- Examined distribution of categorical features such as `jobedu`, `marital`, and `housing`.
- Compared campaign responses across these groups using stacked bar charts.

### **6. Temporal Analysis**
- Created a `date` variable and extracted month and weekday insights.
- Visualized campaign performance trends across months and days.
- Analyzed conversion rates over time and identified high-performing periods.

### **7. Feature Engineering**
Developed new derived features:
- `age_group`: Categorized age into bins (18–24, 25–34, 35–44, etc.)
- `income_category`: Based on salary ranges.
- `balance_category`: Based on account balance.
- `has_loan`, `has_defaulted`, `previous_success`: Binary indicators.
- `customer_score`: Custom metric combining multiple customer factors.

### **8. Correlation Analysis**
- Examined relationships between numeric variables using **heatmaps**.
- Calculated **VIF (Variance Inflation Factor)** to detect multicollinearity.
- Analyzed correlation between categorical encodings and target response.

### **9. Outlier Detection & Treatment**
- Detected outliers using **boxplots**.
- Applied **1st and 99th percentile capping** to treat extreme values.
- Visualized post-treatment distributions using histograms and KDE plots.

---

## 📊 Key Insights
- **Duration of call** has a strong positive correlation with term deposit subscription.  
- Customers with **higher account balances** and **no personal loans** were more likely to subscribe.  
- **Older age groups (45–55)** showed higher response rates compared to younger ones.  
- **Past campaign success** (`previous_success`) strongly influenced current subscription likelihood.  
- Weekdays like **Tuesday and Thursday** saw higher conversion rates.
  
<img width="1185" height="651" alt="image" src="https://github.com/user-attachments/assets/c955509e-45f7-46a7-a229-6902edce60e4" />
<img width="1185" height="651" alt="Screenshot 2025-10-05 152156" src="https://github.com/user-attachments/assets/a82f6c8b-70ca-409f-97f7-c6a8cfc6bb30" />
<img width="1191" height="643" alt="Screenshot 2025-10-05 152116" src="https://github.com/user-attachments/assets/5d685fd1-a1b2-43f2-b46f-10638ee3e0eb" />
<img width="1166" height="700" alt="Screenshot 2025-10-05 152050" src="https://github.com/user-attachments/assets/3f56166e-1587-49ca-bf72-0d38b2365984" />



---

## 🧮 Statistical Techniques Used
- Descriptive statistics (mean, median, std)
- Correlation analysis
- Outlier detection (IQR, percentile capping)
- Data encoding and normalization
- Visualization-based inference

## 🏁 Conclusion
The project provides actionable insights into customer behavior and campaign efficiency, guiding better targeting strategies.
Using Python, this analysis demonstrates how data-driven insights can significantly improve marketing outcomes
