# Case_Study_On_Bank_Telemarketig_campaign
 Python_ML
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime
import re
import sklearn.preprocessing as StandardScaler
import sklearn.linear_model as LinearRegression
import sklearn.metrics as mean_squared_error
import warnings
warnings.filterwarnings("ignore")

#### 1. Loading the Dataset

# a. Load and scrutinize the dataset to comprehend its structure, encompassing columns and data types.
df = pd.read_csv(r"C:\Users\husai\OneDrive\Desktop\Project&Assignments\Python_Project_graded_2\bank_marketing_updated_v1.csv")
df.head()

# data frame with Meaning full Header
new_header = df.iloc[1]
df = df[2:]
df.columns = new_header
df.head()

#### 1. Understanding the Dataset
- a. Load and scrutinize the dataset to comprehend its structure, encompassing columns and data types.

df.info()

df.isna().sum()

df['month'].mode()

# Option 1: Fill missing months (if you know a specific value)
df['month'].fillna('may, 2017', inplace=True)

df.dropna(inplace=True)
df.isna().sum()

# Ensure 'day' is treated as a string and 'month' has no commas
df['date'] = df['day'].astype(str) + ' ' + df['month'].str.replace(',', '')
df['date'].isna().sum()


# converting "duration" in single unit
def convert_to_seconds(time_str):
    total_seconds = 0
    
    # Regex to find minutes and seconds
    min_match = re.search(r'(\d+(\.\d+)?)\s*min', time_str)
    sec_match = re.search(r'(\d+)\s*sec', time_str)
    
    if min_match:
        # Convert minutes to seconds
        total_seconds += float(min_match.group(1)) * 60
    
    if sec_match:
        # Convert seconds
        total_seconds += int(sec_match.group(1))
    
    return total_seconds

# Apply the function to the DataFrame
df['duration_sec'] = df['duration'].apply(convert_to_seconds)

df.head()

df.duplicated().sum()



# Checking the data type of all variable
df.info()

- b. Inspect for any instances of missing values, outliers, or data inconsistencies.

# Convert columns to appropriate types
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
df['pdays'] = pd.to_numeric(df['pdays'], errors='coerce')
df['previous'] = pd.to_numeric(df['previous'], errors='coerce')
df['duration_sec'] = df['duration_sec'].astype(float)

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df.info()

# Create separate DataFrames
df1 = df[numeric_cols]  # DataFrame for numeric columns
df2 = df[categorical_cols]  # DataFrame for categorical columns

# Convert categorical columns to category type
df2[categorical_cols] = df2[categorical_cols].astype('category')

# Display the results
df1.head()

df2.head()

#### 2. Descriptive Statistics
- a. Derive summary statistics (mean, median, standard deviation) for relevant columns.

# 2a. summary statistics (mean, median, standard deviation) 
df1.describe().T

- b. Examine the distribution of the target variable.

distribution = df['targeted'].value_counts(normalize=True)
print("Distribution of targeted:")
print(distribution)

# Plot the bar plot of distribution of term diposite
sns.countplot(x='response', data=df, order=df['response'].value_counts().index)
plt.title('Distribution of Term Deposit Subscriptions')
plt.xlabel('Subscription Status')
plt.ylabel('Count')
plt.show()

#### 3. Univariate Analysis
- a. Examine the distribution of individual key features, such as age, balance, and call duration.

df1[['age','duration_sec','balance']].describe().T

- b. Employ visual aids like histograms, box plots, and kernel density plots to discern patterns and outliers.Set up the matplotlib figure
- Box Plot, Histogram, Density Plot.

#Box Plot
plt.figure(figsize=(18, 6))

# Box plot for age
plt.subplot(1, 3, 1)
sns.boxplot(y=df['age'], color='skyblue')
plt.title('Age Box Plot')
plt.ylabel('Age')

# Box plot for balance
plt.subplot(1, 3, 2)
sns.boxplot(y=df['balance'], color='lightgreen')
plt.title('Balance Box Plot')
plt.ylabel('Balance')

# Box plot for call duration
plt.subplot(1, 3, 3)
sns.boxplot(y=df['duration_sec'], color='salmon')
plt.title('Call Duration Box Plot')
plt.ylabel('Call Duration (seconds)')

plt.tight_layout()
plt.show()


#Histogram

plt.figure(figsize=(18, 6))

# Histogram for age
plt.subplot(1, 3, 1)
sns.histplot(df['age'], bins=10, kde=False, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Histogram for balance
plt.subplot(1, 3, 2)
sns.histplot(df['balance'], bins=10, kde=False, color='lightgreen')
plt.title('Balance Distribution')
plt.xlabel('Balance')
plt.ylabel('Frequency')

# Histogram for call duration
plt.subplot(1, 3, 3)
sns.histplot(df['duration_sec'], bins=10, kde=False, color='salmon')
plt.title('Call Duration Distribution')
plt.xlabel('Call Duration (seconds)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Set up the matplotlib figure
plt.figure(figsize=(18, 6))

# KDE plot for age
plt.subplot(1, 3, 1)
sns.kdeplot(df['age'], fill=True, color='skyblue')
plt.title('Age KDE Plot')
plt.xlabel('Age')
plt.ylabel('Density')

# KDE plot for balance
plt.subplot(1, 3, 2)
sns.kdeplot(df['balance'], fill=True, color='lightgreen')
plt.title('Balance KDE Plot')
plt.xlabel('Balance')
plt.ylabel('Density')

# KDE plot for call duration
plt.subplot(1, 3, 3)
sns.kdeplot(df['duration_sec'], fill=True, color='salmon')
plt.title('Call Duration KDE Plot')
plt.xlabel('Call Duration (seconds)')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

#### 4. Bivariate Analysis
- a. Evaluate the relationship between independent variables and the target variable.

# Encode the target variable as binary
df['response'] = df['response'].map({'yes': 1, 'no': 0})

# Select numerical independent variables
numerical_vars = ['age', 'salary', 'balance', 'pdays', 'previous', 'duration_sec']

# Calculate correlation with the target variable
correlation_matrix = df[numerical_vars + ['response']].corr()
print("Correlation with target variable:")
print(correlation_matrix['response'].sort_values(ascending=False))

# Categorical variables
categorical_vars = ['marital', 'jobedu', 'targeted', 'default', 'housing', 'loan', 'poutcome']

# Scatter plots for numerical features against the target variable
for col in numerical_vars:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=col, y='response', alpha=0.5)
    plt.title(f'Scatter Plot of {col} vs. Response')
    plt.xlabel(col)
    plt.ylabel('Response')
    plt.show()

    
## Box plots for categorical variables
for col in categorical_vars:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=col, y='response')
    plt.title(f'Box Plot of {col} vs. Response')
    plt.xlabel(col)
    plt.ylabel('Response')
    plt.show()

    
# Bar plots for mean response by categorical variable
for col in categorical_vars:
    mean_response = df.groupby(col)['response'].mean()
    mean_response.plot(kind='bar', figsize=(8, 5))
    plt.title(f'Mean Response by {col}')
    plt.xlabel(col)
    plt.ylabel('Mean Response')
    plt.show()
    

# Select relevant columns 
selected_columns = ['age', 'balance', 'duration_sec', 'response'] 

# Pairplot to visualize relationships 
sns.pairplot(df[selected_columns], hue='response', palette='coolwarm') 
plt.suptitle('Relationships between Independent Variables and Response', y=1.02) 
plt.show()

- b. Analyze how features like age, job type, education, marital status, etc. associate with the success of the term deposit campaign, using visualizations like bar charts, stacked bar charts, and heat maps.

##  the average response rate based on categorical features such as job, education, and marital status.

# Bar Chart for Job Type
plt.figure(figsize=(8, 4))
job_response = df.groupby('jobedu')['response'].mean().sort_values()
job_response.plot(kind='bar', color='skyblue')
plt.title('Average Response Rate by Job Type')
plt.xlabel('Job Type')
plt.ylabel('Average Response Rate')
plt.xticks(rotation=90)
plt.show()

# Bar chart for education targeted
plt.figure(figsize=(8, 4))
sns.countplot(x='salary', hue='response', data=df)
plt.title('Education Level vs response')
plt.xlabel('Education Level')
plt.ylabel('Count')

# Bar Chart for Marital Status
plt.figure(figsize=(8, 4))
marital_response = df.groupby('marital')['response'].mean().sort_values()
marital_response.plot(kind='bar', color='lightgreen')
plt.title('Average Response Rate by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Average Response Rate')
plt.xticks(rotation=45)
plt.show()

# Bar chart for age vs targeted (bins for better visualization)
plt.figure(figsize=(8, 4))
df['age_bin'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 60], labels=['20-30', '30-40', '40-50', '50-60'])
sns.countplot(x='age_bin', hue='response', data=df)
plt.title('Response by Age Range' )
plt.xlabel('Age Range')
plt.ylabel('Count')

#Stacked bar charts can show the relationship between two categorical variables, such as marital status and response
marital_counts = pd.crosstab(df['marital'], df['response'])
marital_counts.plot(kind='bar', stacked=True, color=['orange', 'lightblue'], figsize=(8, 4))
plt.title('Response by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(['No', 'Yes'], title='Response')
plt.show()


# Calculate correlations
correlation_matrix = df1.corr()

# Generate a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

### 5. Categorical Variables Analysis
- a. Investigate the distribution of categorical variables such as job type, education, and marital status.

# Analyze the distribution of 'jobedu' (job type and education)
jobedu_distribution = df['jobedu'].value_counts()
print("Job and Education Distribution:\n", jobedu_distribution)

# Analyze the distribution of 'marital' status
marital_distribution = df['marital'].value_counts()
print("\nMarital Status Distribution:\n", marital_distribution)

# Filter the dataset to include only the top 15 jobedu combinations 
# Calculate the top 15 jobedu combinations 
top_15_jobedu = df['jobedu'].value_counts().nlargest(15).index
df_top_15 = df[df['jobedu'].isin(top_15_jobedu)]

# Plot the distribution 
plt.figure(figsize=(10, 6))
sns.countplot(y='jobedu', data=df_top_15, order=top_15_jobedu)
plt.title('Distribution of Top 15 Job Type and Education Combinations')
plt.xlabel('Count')
plt.ylabel('Job Type and Education')
plt.show()

# Plot for marital status distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='marital', data=df, order=df['marital'].value_counts().index)
plt.title('Distribution of Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()


- b. Assess the impact of these categorical variables on the campaign's success through visualizations like bar charts.

df['response'] = df['response'].map({1: 'yes', 0: 'no'})

# Assuming 'response' is the target variable indicating campaign success

# Bar chart for jobedu and campaign response
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='jobedu', hue='response', palette='Set1', order=df['jobedu'].value_counts().index)
plt.title('Campaign Response by Job Type and Education')
plt.xlabel('Job Type and Education')
plt.ylabel('Count')
plt.legend(title='Response', loc='upper right', labels=['No', 'Yes'])
plt.xticks(rotation=90)
plt.show()

# Bar chart for marital status and campaign response
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='marital', hue='response', palette='Set1')
plt.title('Campaign Response by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.legend(title='Response', loc='upper right', labels=['No', 'Yes'])
plt.show()


#### 6. Temporal Analysis
- a. Investigate temporal patterns in the success of the campaign over time.

# Attempt to convert with a specific format if known
df['month'] = pd.to_datetime(df['month'])  # Adjust format as needed

MonthYear_counts = df['month'].value_counts().sort_index()

# Plotting
plt.figure(figsize=(12, 6))
MonthYear_counts.plot(kind='bar', color='skyblue')
plt.title('Count of Entries for Each Unique Date')
plt.xlabel('Date')
plt.ylabel('Count of Entries')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

MonthYear_counts 

# Calculate success metrics
campaign_results = df.groupby('month').agg(
    total_contacts=('customerid', 'count'),
    successful_responses=('response', lambda x: (x == 'yes').sum())
).reset_index()

# Calculate conversion rate
campaign_results['conversion_rate'] = campaign_results['successful_responses'] / campaign_results['total_contacts']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(campaign_results['month'].astype(str), campaign_results['conversion_rate'], marker='o')
plt.title('Campaign Conversion Rate Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Conversion Rate')
plt.xticks(rotation=45)
plt.grid()
plt.show()

campaign_results['conversion_rate']

- b. Analyze if specific months or days exhibit superior campaign performance.

#Convert 'date' column to datetime.
df['date'] = pd.to_datetime(df['date'], format='%d %b %Y', errors='coerce')

 # Extracts day of the week
df['day_of_week'] = df['date'].dt.day_name()
df['day_of_week']

# Group by day of the week and calculate the conversion rate
day_conversion_rate = df.groupby('day_of_week')['response'].apply(lambda x: (x == 'yes').mean()).sort_values(ascending=False)

# Display the result
print(day_conversion_rate)

# Plot conversion rate by day of the week
plt.figure(figsize=(10, 6))
day_conversion_rate.plot(kind='bar', color='salmon')
plt.title('Conversion Rate by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Conversion Rate')
plt.xticks(rotation=45)
plt.show()

### 7. Feature Engineering
- a. Introduce new features that may enhance prediction, such as creating age groups or income categories.

# Define age groups
def age_group(age):
    if age < 25:
        return '18-24'
    elif age < 35:
        return '25-34'
    elif age < 45:
        return '35-44'
    elif age < 55:
        return '45-54'
    else:
        return '55+'

# Create a new column for age groups
df['age_group'] = df['age'].apply(age_group)
df['age_group']

# Define income categories
def income_category(salary):
    if salary <= 30000:
        return 'Low'
    elif salary <= 70000:
        return 'Medium'
    elif salary <= 150000:
        return 'High'
    else:
        return 'Very High'

# Create a new column for income categories
df['income_category'] = df['salary'].apply(income_category)
df['income_category']

# Define balance categories
def balance_category(balance):
    if balance < 500:
        return 'Low'
    elif balance < 2000:
        return 'Medium'
    else:
        return 'High'

# Create a new column for balance categories
df['balance_category'] = df['balance'].apply(balance_category)
df['balance_category']

# Create a binary feature for loan status
df['has_loan'] = df['loan'].apply(lambda x: 1 if x == 'yes' else 0)
df['has_loan']

# Create a binary feature for default status
df['has_defaulted'] = df['default'].apply(lambda x: 1 if x == 'yes' else 0)
df['has_defaulted']

# Create a feature indicating if a customer responded to previous campaigns
df['previous_success'] = df['previous'].apply(lambda x: 1 if x > 0 else 0)
df['previous_success']

# Example: create a customer score based on multiple factors
df['customer_score'] = df['age'] / 10 + df['salary'] / 10000 + df['balance'] / 1000
df['customer_score']

# Convert 'response' to numeric (1 for 'yes', 0 for 'no')
df['response_numeric'] = df['response'].apply(lambda x: 1 if x == 'yes' else 0)
df['response_numeric']



# Create pair plots for new features
sns.pairplot(df, hue='response_numeric', vars=['age', 'salary', 'balance', 'has_loan', 'has_defaulted'])
plt.title('Pair Plot of New Features with Response')
plt.show()


- b. Apply encoding techniques to transform categorical variables if necessary.

df_cat = df2[['marital', 'jobedu', 'targeted', 'default', 'housing', 'loan', 'contact']]

pd.get_dummies(df_cat).head()

#### 8. Correlation Analysis
- a. Examine correlations between independent variables to identify multicollinearity.

# List of numeric columns you want to check for correlations
numeric_columns = [
    'age', 'salary', 'balance', 'pdays', 'previous', 'duration_sec', 
    'customer_score', 'has_loan', 'has_defaulted', 'previous_success'
]



# All numerical columns
df1 = df[['age', 'salary', 'balance', 'pdays', 'previous', 'duration_sec', 'customer_score', 'has_loan', 'has_defaulted', 'previous_success']]

# Compute the correlation matrix for the numeric columns
correlation_matrix = df1.corr()

# Display the correlation matrix
print(correlation_matrix)

# Plot the heatmap for better visualization
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numeric Variables')
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Add a constant column to the numeric data (for intercept in VIF calculation)
X = df1
X = add_constant(X)

# Compute the VIF for each variable
vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display VIF results
print(vif_data)


- b. Evaluate how correlated features may influence the target variable.

# selecting the catagorical columns
df2 = df[['marital', 'jobedu', 'targeted', 'default', 'housing', 'loan', 'contact', 'poutcome', 'day_of_week', 'age_group']]
df2.head()

# Initialize label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in df2:
    df[col] = le.fit_transform(df[col])
    

# Calculate the correlation matrix for the DataFrame
corr_matrix = df2.corr()
corr_matrix

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

#### 9. Outlier Detection and Handling
- a. Identify and rectify outliers that could impact the analysis and predictions.


 #List of numeric columns to check for outliers
numeric_columns = df.select_dtypes(include=['number', 'float64', 'int64']).columns
numeric_columns

# Calculate number of rows and columns for the subplots
num_columns = 4  # You can adjust this number
num_rows = (len(numeric_columns) // num_columns) + (1 if len(numeric_columns) % num_columns != 0 else 0)


# Create subplots
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(num_rows, num_columns, i)  # Adjusted to handle variable number of columns/rows
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# Cap the values at the 1st and 99th percentiles
for col in numeric_columns:
    lower_percentile = df[col].quantile(0.01)
    upper_percentile = df[col].quantile(0.99)
    
    # Cap values below the 1st percentile to the 1st percentile value
    # and values above the 99th percentile to the 99th percentile value
    df[col] = np.clip(df[col], lower_percentile, upper_percentile)

# Check the data after capping
df.head()

# Histogram for 'salary'
plt.subplot(1, 3, 1)
sns.histplot(df['salary'], bins=10, kde=False, color='skyblue')
plt.title('salary Distribution')
plt.xlabel('salary')
plt.ylabel('Frequency')

# KDE plot for 'salary'
plt.subplot(1, 3, 3)
sns.kdeplot(df['salary'], fill=True, color='salmon')
plt.title('salary KDE Plot')
plt.xlabel('salary (seconds)')
plt.ylabel('Density')



