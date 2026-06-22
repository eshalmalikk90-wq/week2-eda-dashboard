import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Performance EDA Dashboard", layout="wide")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_student_data.csv")
    return df

df = load_data()

# -------------------------
# TITLE
# -------------------------
st.title("📊 Student Performance EDA Dashboard")
st.markdown("Interactive analysis of student performance dataset")

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("Filters")

# Example categorical filter (adjust column name if needed)
categorical_cols = df.select_dtypes(include='object').columns.tolist()

selected_cat_col = None
selected_values = None

if categorical_cols:
    selected_cat_col = st.sidebar.selectbox("Choose Category Column", categorical_cols)
    selected_values = st.sidebar.multiselect(
        "Filter Values",
        df[selected_cat_col].dropna().unique()
    )

if selected_cat_col and selected_values:
    df = df[df[selected_cat_col].isin(selected_values)]

# -------------------------
# DATA OVERVIEW
# -------------------------
st.header("📌 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

st.write("### Column Info")
st.dataframe(df.dtypes.astype(str))

st.write("### Sample Data")
st.dataframe(df.head())

# -------------------------
# DATA CLEANING SUMMARY
# -------------------------
st.header("🧹 Data Cleaning Summary")

st.write("Missing Values per Column")
st.dataframe(df.isnull().sum())

st.write("Duplicate Rows:", df.duplicated().sum())

# -------------------------
# STATISTICAL SUMMARY
# -------------------------
st.header("📊 Statistical Summary")
st.dataframe(df.describe())

# -------------------------
# VISUALIZATION SECTION
# -------------------------
st.header("📈 Exploratory Data Analysis")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Histogram
st.subheader("Distribution Analysis")

selected_hist = st.selectbox("Select Column for Histogram", numeric_cols)

fig, ax = plt.subplots()
sns.histplot(df[selected_hist], kde=True, ax=ax)
st.pyplot(fig)

# Boxplot
st.subheader("Outlier Detection")

selected_box = st.selectbox("Select Column for Boxplot", numeric_cols)

fig, ax = plt.subplots()
sns.boxplot(x=df[selected_box], ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Analysis")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -------------------------
# INSIGHTS SECTION
# -------------------------
st.header("💡 Insights")

st.write("""
- You can identify distribution patterns from histograms  
- Boxplots show outliers in student performance features  
- Correlation heatmap shows relationships between variables  
- Filters help analyze sub-groups of students  
""")