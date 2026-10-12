import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# ----------------------------------------------------
# 1. Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="AI Data Analytics Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Sales Analytics & Insights Dashboard")
st.markdown("Upload your transactional data or use generated data to run automated EDA, AI Segmentation, and Executive Summaries.")

# ----------------------------------------------------
# 2. Data Loader / Generator
# ----------------------------------------------------
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 300
    dates = pd.date_range(start="2026-01-01", periods=n, freq="D")
    categories = ["Electronics", "Fitness Gear", "Clothing", "Home Appliances"]
    
    df = pd.DataFrame({
        "TransactionDate": dates,
        "Category": np.random.choice(categories, n),
        "SalesAmount": np.random.uniform(20, 500, n).round(2),
        "UnitsSold": np.random.randint(1, 10, n),
        "CustomerRating": np.random.uniform(2.5, 5.0, n).round(1)
    })
    return df

st.sidebar.header("Data Options")
data_source = st.sidebar.radio("Data Source:", ("Use Sample Retail Dataset", "Upload CSV File"))

if data_source == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Awaiting CSV file. Loading sample retail dataset for now...")
        df = load_sample_data()
else:
    df = load_sample_data()

# ----------------------------------------------------
# 3. Key Metrics KPI Cards
# ----------------------------------------------------
st.subheader("📌 Key Business Metrics")
col1, col2, col3, col4 = st.columns(4)

total_revenue = df["SalesAmount"].sum()
total_orders = len(df)
avg_rating = df["CustomerRating"].mean()
top_category = df.groupby("Category")["SalesAmount"].sum().idxmax()

col1.metric("Total Revenue", f"${total_revenue:,.2f}")
col2.metric("Total Transactions", f"{total_orders:,}")
col3.metric("Avg Customer Rating", f"{avg_rating:.2f} / 5.0")
col4.metric("Top Performing Category", top_category)

st.markdown("---")

# ----------------------------------------------------
# 4. Interactive Visualizations
# ----------------------------------------------------
st.subheader("📈 Interactive Exploratory Analytics")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_cat = px.bar(
        df.groupby("Category", as_index=False)["SalesAmount"].sum(),
        x="Category", y="SalesAmount",
        color="Category",
        title="Revenue Distribution by Category"
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with chart_col2:
    if "TransactionDate" in df.columns:
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
        time_df = df.groupby("TransactionDate")["SalesAmount"].sum().reset_index()
        fig_line = px.line(
            time_df, x="TransactionDate", y="SalesAmount",
            title="Sales Trend Over Time"
        )
        st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------
# 5. ML / AI Feature: Automated Customer Clustering
# ----------------------------------------------------
st.subheader("🤖 AI Customer Behavior Segmentation")

# Prepare feature matrix
X = df[["SalesAmount", "UnitsSold"]].dropna()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

fig_cluster = px.scatter(
    df, x="SalesAmount", y="UnitsSold",
    color=df["Cluster"].astype(str),
    title="Customer Purchase Clusters (K-Means AI)",
    labels={"color": "Segment"}
)
st.plotly_chart(fig_cluster, use_container_width=True)

# ----------------------------------------------------
# 6. Automated Natural Language Executive Summary
# ----------------------------------------------------
st.subheader("📝 AI Executive Summary & Recommendations")

cat_revenue = df.groupby("Category")["SalesAmount"].sum().sort_values(ascending=False)
lowest_cat = cat_revenue.index[-1]

insight_text = f"""
> **Automated Data Insights:**
> * **Revenue Driver:** `{cat_revenue.index[0]}` generated the highest revenue, contributing **${cat_revenue.iloc[0]:,.2f}**.
> * **Growth Opportunity:** `{lowest_cat}` registered the lowest total sales (**${cat_revenue.iloc[-1]:,.2f}**). Consider targeted promotional campaigns.
> * **Customer Experience:** Overall satisfaction averages **{avg_rating:.2f}/5.0**. Transactions with ratings below 3.0 should be flagged for operational review.
"""
st.markdown(insight_text)

# Data Table Display
with st.expander("🔍 View Raw Dataset"):
    st.dataframe(df)