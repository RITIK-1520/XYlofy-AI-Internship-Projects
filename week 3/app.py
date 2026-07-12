import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

PRIMARY = "#4F8BF9"
SECONDARY = "#00C49F"
ACCENT = "#FFB347"
BACKGROUND = "#111827"


st.set_page_config(

    page_title="Sales Forecasting Dashboard",

    page_icon="📈",

    layout="wide"
)
st.title(
    "Sales Forecasting & Demand Intelligence System"
)

st.caption(
    "Retail Sales Forecasting | Anomaly Detection | Product Demand Segmentation"
)


@st.cache_data
def load_data():

    sales = pd.read_csv(
        "superstore_sales.csv"
    )

    sales["Order Date"] = pd.to_datetime(
        sales["Order Date"],
        dayfirst=True
    )

    sales["Year"] = sales["Order Date"].dt.year

    sales["Month"] = sales["Order Date"].dt.month_name()

    return sales

sales = load_data()
with st.spinner("Loading Dashboard..."):
    sales = load_data()

st.sidebar.title("Welcome to the Dashboard")

page = st.sidebar.radio(
    "",
[ "🏠 Home",
  "📈 Forecast Explorer",
    "🚨 Anomaly Report",
    "📦 Demand Segments",
    "ℹ️ About" ]

   # Page 1
)
if page=="🏠 Home":
    st.header("📊 Sales Overview Dashboard")

    st.write(
        "This dashboard provides an overview of retail sales performance, trends, and business insights."
    )

    st.markdown("---")

    total_sales = sales["Sales"].sum()

    total_orders = sales["Order ID"].nunique()

    total_categories = sales["Category"].nunique()

    total_regions = sales["Region"].nunique()


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric( "💰 Total Sales",
                f"${total_sales/1000000:.2f} M"
        )

    with col2:
        st.metric("📦 Orders",
                total_orders
        )

    with col3:
        st.metric("🛒 Categories",
                total_categories
        )

    with col4:
        st.metric("🌍 Regions",
                total_regions
        )

    yearly_sales = sales.groupby("Year")["Sales"].sum().reset_index()

    fig = px.bar(yearly_sales,x="Year", y="Sales",text_auto=".2s", title="📈 Total Sales by Year")

    fig.update_traces(
        marker_color=PRIMARY
    )

    fig.update_layout(

        template="plotly_dark",font=dict(size=16),
        margin=dict(l=20,  r=20,t=50, b=20))

    st.plotly_chart(fig,use_container_width=True)


    monthly_sales = sales.groupby(pd.Grouper( key="Order Date", freq="ME"))["Sales"].sum().reset_index()

    fig = px.line(monthly_sales,x="Order Date",y="Sales",markers=True,title="📅 Monthly Sales Trend")

    fig.update_layout( title_x=0.25)

    st.plotly_chart( fig, use_container_width=True)

    left, right = st.columns(2)

    with left:

        selected_region = st.selectbox("🌍 Select Region",["All"] + list(sales["Region"].unique()))

    with right:

        selected_category = st.selectbox("🛒 Select Category",["All"] + list(sales["Category"].unique() ))

    filtered_sales = sales.copy()

    if selected_region != "All":filtered_sales = filtered_sales[ filtered_sales["Region"] == selected_region ]


    if selected_category != "All":filtered_sales = filtered_sales[ filtered_sales["Category"] == selected_category]

    st.subheader("Filtered Sales Data")

    st.dataframe(filtered_sales,use_container_width=True)
    top_category = (
        sales.groupby("Category")["Sales"]
        .sum()
        .idxmax()
    )

    st.success(
    f"""
    📌Business Insights

    **{top_category}** contributes the highest revenue.

    Managers should prioritize inventory for this category.
    """
    )

# PAGE 2 : FORECAST EXPLORER

elif page == "📈 Forecast Explorer":

    st.header("📈 Forecast Explorer")
    st.write("Forecast future sales using the best performing model.")

    comparison = pd.read_csv("comparison_table.csv")
    forecast = pd.read_csv("segment_forecast.csv", index_col=0)

    col1, col2 = st.columns(2)

    with col1:
        segment = st.selectbox(
            "Select Category / Region",
            forecast.columns
        )

    with col2:
        horizon = st.slider(
            "Forecast Horizon (Months)",
            1, 3, 3
        )

    best = comparison.sort_values("MAPE").iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{best['MAE']:.2f}")
    c2.metric("RMSE", f"{best['RMSE']:.2f}")
    c3.metric("MAPE", f"{best['MAPE']:.2f}%")

    forecast_data = pd.DataFrame({
        "Month": forecast.index[:horizon],
        "Forecast": forecast[segment].values[:horizon]
    })

    st.markdown("---")

        #  Forecast Chart

    fig = px.line(
        forecast_data,
        x="Month",
        y="Forecast",
        markers=True,
        text="Forecast",
        title=f"{segment} Sales Forecast"
    )

    fig.update_traces(
        texttemplate="%{text:.0f}",
        textposition="top center",
        line=dict(width=4)
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        title_x=0.35
    )

    st.plotly_chart(fig, use_container_width=True)

    #  Forecast Table 

    st.subheader("Forecast Values")

    st.dataframe(
        forecast_data.style.format({"Forecast":"{:,.2f}"}),
        use_container_width=True
    )

    # Forecast Summary

    highest = forecast_data.loc[forecast_data["Forecast"].idxmax()]
    lowest = forecast_data.loc[forecast_data["Forecast"].idxmin()]

    growth = (
        (forecast_data["Forecast"].iloc[-1]
        - forecast_data["Forecast"].iloc[0])
        /
        forecast_data["Forecast"].iloc[0]
    ) * 100

    a, b, c = st.columns(3)

    a.metric(
        "Highest Forecast",
        highest["Month"],
        f"{highest['Forecast']:,.0f}"
    )

    b.metric(
        "Lowest Forecast",
        lowest["Month"],
        f"{lowest['Forecast']:,.0f}"
    )

    c.metric(
        "Growth",
        f"{growth:.2f}%"
    )

    #  Forecast Accuracy 

    accuracy = 100 - best["MAPE"]

    st.subheader("Forecast Accuracy")

    st.progress(int(accuracy))

    st.write(
        f"Model Accuracy : **{accuracy:.2f}%**"
    )

    # Best Model 

    st.subheader("Best Forecasting Model")

    st.dataframe(
        comparison.sort_values("MAPE").head(1).round(2),
        use_container_width=True
    )

    # Business Insight

    st.success(f"""
### 📌 Business Recommendation

• **{best['Model']}** achieved the highest forecasting accuracy.

• Highest expected sales occur during **{highest['Month']}**.

• Expected peak sales are **{highest['Forecast']:,.0f}**.

• Managers should prepare inventory before the expected peak demand.
""")

    #  Download

    st.download_button(
        "📥 Download Forecast",
        forecast_data.to_csv(index=False),
        "forecast.csv",
        "text/csv"
    )
    # Page 3 

elif page=="🚨 Anomaly Report":

    st.header("🚨 Sales Anomaly Report")
    st.write("Analyze unusual weekly sales detected using Isolation Forest.")

    anomaly = pd.read_csv("anomaly_points.csv")
    anomaly["Order Date"] = pd.to_datetime(anomaly["Order Date"])

    weekly = sales.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"].sum().reset_index()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Anomalies", len(anomaly))
    c2.metric("Highest Sales", f"{anomaly['Sales'].max():,.0f}")
    c3.metric("Lowest Sales", f"{anomaly['Sales'].min():,.0f}")

    st.markdown("---")

    fig = px.line(weekly, x="Order Date", y="Sales",
                  title="Weekly Sales with Detected Anomalies")

    fig.add_scatter(
        x=anomaly["Order Date"], y=anomaly["Sales"],
        mode="markers",
        marker=dict(color="red", size=10, symbol="x"),
        name="Anomaly"
    )

    fig.update_layout(template="plotly_dark", height=500, title_x=0.25)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomaly[["Order Date","Sales"]]
        .style.format({"Sales":"{:,.2f}"})
        .highlight_max(subset=["Sales"], color="#ff6961"),
        use_container_width=True
    )

    highest = anomaly.loc[anomaly["Sales"].idxmax()]
    lowest = anomaly.loc[anomaly["Sales"].idxmin()]

    a, b = st.columns(2)
    a.metric("Highest Anomaly",
             highest["Order Date"].strftime("%d-%b-%Y"),
             f"{highest['Sales']:,.0f}")

    b.metric("Lowest Anomaly",
             lowest["Order Date"].strftime("%d-%b-%Y"),
             f"{lowest['Sales']:,.0f}")

    with st.expander("📖 Business Interpretation"):

        st.markdown(f"""
### Key Findings

- **{len(anomaly)}** unusual sales weeks were detected.
- Highest anomaly occurred on **{highest['Order Date'].strftime('%d-%b-%Y')}** with sales of **{highest['Sales']:,.0f}**.
- Sales spikes may be caused by festivals, promotions or bulk orders.
- Low sales may occur due to supply shortages or reduced customer demand.

### Recommendation

Investigate these periods before making inventory and supply chain decisions.
""")

    st.download_button(
        "📥 Download Report",
        anomaly.to_csv(index=False),
        "anomaly_report.csv",
        "text/csv"
    )

#  PAGE 4 : DEMAND SEGMENTS

elif page=="📦 Demand Segments":

    st.header("📦 Product Demand Segmentation")
    st.write("Analyze product demand clusters generated using K-Means clustering.")

    segment=pd.read_csv("segment_data.csv")

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Clusters",segment["Cluster"].nunique())
    c2.metric("Products",len(segment))
    c3.metric("Highest Sales",f"{segment['Sales'].max():,.0f}")
    c4.metric("Avg Growth",f"{segment['Growth_Rate'].mean():.2f}%")

    cluster=st.selectbox("Select Demand Cluster",sorted(segment["Cluster_Name"].unique()))
    filtered=segment[segment["Cluster_Name"]==cluster]

    fig=px.scatter(
        segment,
        x="PCA1",
        y="PCA2",
        color="Cluster_Name",
        size="Sales",
        hover_name="Sub-Category",
        hover_data=["Growth_Rate","Sales_Volatility","Average_Order_Value"],
        title="Demand Segmentation using PCA"
    )

    fig.update_layout(template="plotly_dark",height=550,title_x=0.30)
    st.plotly_chart(fig,use_container_width=True)

    cluster_count=segment["Cluster_Name"].value_counts().reset_index()
    cluster_count.columns=["Cluster","Products"]

    fig=px.pie(
        cluster_count,
        names="Cluster",
        values="Products",
        hole=0.55,
        title="Product Distribution Across Clusters"
    )

    fig.update_traces(textinfo="percent+label")
    fig.update_layout(template="plotly_dark",height=430,title_x=0.25)

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("📊 Cluster Summary")

    st.dataframe(
        filtered[["Sales","Growth_Rate","Sales_Volatility","Average_Order_Value"]]
        .describe()
        .round(2),
        use_container_width=True
    )

    st.subheader("📦 Cluster Members")

    st.dataframe(
        filtered[
            [
                "Sub-Category",
                "Sales",
                "Growth_Rate",
                "Sales_Volatility",
                "Average_Order_Value"
            ]
        ].round(2),
        use_container_width=True
    )

    top=filtered.loc[filtered["Sales"].idxmax()]

    c1,c2,c3=st.columns(3)
    c1.metric("Products",len(filtered))
    c2.metric("Top Product",top["Sub-Category"])
    c3.metric("Average Sales",f"{filtered['Sales'].mean():,.0f}")

    top5=filtered.sort_values("Sales",ascending=False).head(5)

    fig=px.bar(
        top5,
        x="Sub-Category",
        y="Sales",
        color="Sales",
        text="Sales",
        title="Top 5 Products"
    )

    fig.update_traces(texttemplate="%{text:.0f}",textposition="outside")
    fig.update_layout(template="plotly_dark",height=450,showlegend=False)

    st.plotly_chart(fig,use_container_width=True)

    strategy={
        "High Value, Rapid Growth":"Increase inventory and prioritize replenishment.",
        "Growing Demand":"Gradually increase stock to meet future demand.",
        "High Volume, High Demand":"Maintain high safety stock with frequent replenishment.",
        "Low Volume, Stable Demand":"Maintain moderate inventory and avoid overstocking."
    }

    with st.expander("📖 Business Insight & Stocking Strategy"):

        st.markdown(f"""
### 📌 Key Findings

- **Selected Cluster:** {cluster}
- **Products:** {len(filtered)}
- **Top Product:** {top['Sub-Category']}
- **Highest Sales:** {top['Sales']:,.0f}
- **Average Growth:** {filtered['Growth_Rate'].mean():.2f}%

### 📦 Recommended Strategy

{strategy.get(cluster)}

### 📈 Business Recommendation

Maintain inventory based on the demand characteristics of this cluster. Frequently replenish high-demand products while avoiding excess inventory for low-demand products.
""")

    st.download_button(
        "📥 Download Cluster Report",
        filtered.to_csv(index=False),
        "cluster_report.csv",
        "text/csv"
    )

    # PAGE 5 

elif page=="ℹ️ About":

    st.header("ℹ️ About Project")

    st.markdown("""
## 🛒 Retail Sales Forecasting & Demand Segmentation Dashboard

This dashboard analyzes historical retail sales data and provides insights for forecasting future sales, detecting anomalies, and identifying product demand segments using Machine Learning.

### 🎯 Project Objectives

- Forecast future sales
- Detect unusual sales patterns
- Segment products based on demand
- Support inventory planning
- Generate business insights

### 🤖 Machine Learning Models

- SARIMA
- Prophet
- XGBoost
- Isolation Forest
- K-Means Clustering
- PCA

### 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Statsmodels
- Prophet
- XGBoost
- Plotly
- Streamlit

### 📊 Dashboard Features

- Sales Overview
- Forecast Explorer
- Anomaly Detection
- Demand Segmentation
- Business Recommendations
- Downloadable Reports
""")

    c1,c2,c3=st.columns(3)

    c1.metric("Dataset","Superstore Sales")
    c2.metric("Forecast Models","3")
    c3.metric("Dashboard Pages","5")

    st.success("""
### 📌 Business Impact

This dashboard helps managers forecast demand, optimize inventory, identify unusual sales trends, and make data-driven business decisions.
""")

    st.info("""
👨‍💻 Developed as part of the AI/ML Internship Project using Python and Streamlit.
""")