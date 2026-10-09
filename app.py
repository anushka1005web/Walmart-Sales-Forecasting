import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F7F9FC;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

.navbar {
    background: #071A3D;
    padding: 18px 30px;
    border-radius: 14px;
    margin-bottom: 25px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.nav-logo {
    color: white;
    font-size: 22px;
    font-weight: 800;
}

.nav-text {
    color: #CBD5E1;
    font-size: 14px;
}

.hero {
    background: linear-gradient(135deg, #071A3D 0%, #0B2A63 60%, #124A9E 100%);
    padding: 65px 55px;
    border-radius: 24px;
    color: white;
    margin-bottom: 30px;
}

.hero-small {
    color: #93C5FD;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 15px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 18px;
}

.hero-subtitle {
    font-size: 19px;
    color: #DCEBFF;
    max-width: 720px;
    line-height: 1.6;
}

.hero-description {
    font-size: 15px;
    color: #BFD4F2;
    max-width: 720px;
    line-height: 1.7;
    margin-top: 15px;
}

.graph-box {
    margin-top: 35px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 18px;
}

.graph-line {
    color: #60A5FA;
    font-size: 30px;
    letter-spacing: 4px;
}

.section-title {
    color: #071A3D;
    font-size: 30px;
    font-weight: 800;
    margin-top: 35px;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #64748B;
    font-size: 15px;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
    height: 100%;
}

.card-icon {
    font-size: 28px;
    margin-bottom: 12px;
}

.card-title {
    color: #071A3D;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
}

.card-text {
    color: #64748B;
    font-size: 14px;
    line-height: 1.6;
}

.kpi {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 5px 18px rgba(15,23,42,0.05);
}

.kpi-value {
    color: #0B3B82;
    font-size: 27px;
    font-weight: 800;
}

.kpi-label {
    color: #64748B;
    font-size: 13px;
    margin-top: 5px;
}

.flow {
    background: #EFF6FF;
    border: 1px solid #DBEAFE;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    color: #0B3B82;
    font-weight: 700;
}

.footer {
    margin-top: 55px;
    background: #071A3D;
    color: #CBD5E1;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
}

.footer-title {
    color: white;
    font-size: 18px;
    font-weight: 700;
}

@media (max-width: 700px) {
    .hero {
        padding: 40px 25px;
    }
    .hero-title {
        font-size: 34px;
    }
    .navbar {
        padding: 15px 18px;
    }
}
</style>
""")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Unable to load train.csv")
    st.write(e)
    st.stop()

# Load trained XGBoost model
xgb_model = joblib.load("xgboost_model.pkl")
xgb_features = joblib.load("xgboost_features.pkl")

# --------------------------------------------------
# NAVBAR
# --------------------------------------------------

st.html("""
<div class="navbar">
    <div class="nav-logo">
        📈 Walmart Forecasting
    </div>

    <div class="nav-text">
        Time Series Analysis &nbsp; • &nbsp; Machine Learning
    </div>
</div>
""")

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.html("""
<div class="hero">
    <div class="hero-small">
        DATA ANALYTICS • TIME SERIES • MACHINE LEARNING
    </div>

    <div class="hero-title">
        WALMART SALES<br>
        FORECASTING
    </div>

    <div class="hero-subtitle">
        Predicting Retail Sales Using Time Series Analysis
        &amp; Machine Learning
    </div>

    <div class="hero-description">
        An interactive forecasting solution that analyzes
        historical Walmart sales data and provides insights
        into future retail demand using SARIMA and XGBoost.
    </div>

    <div class="graph-box">
        <div class="graph-line">
            ╱╲___╱╲____╱╲___╱╲____╱╲
        </div>
        <div style="color:#94A3B8; font-size:12px;">
            Historical Sales → Pattern Analysis → Forecast
        </div>
    </div>
</div>
""")

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

st.html("""
<div class="section-title">Project at a Glance</div>
<div class="section-subtitle">
    Key information from the Walmart retail sales dataset
</div>
""")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.html(f"""
    <div class="kpi">
        <div class="kpi-value">{len(df):,}</div>
        <div class="kpi-label">Sales Records</div>
    </div>
    """)

with c2:
    st.html(f"""
    <div class="kpi">
        <div class="kpi-value">{df["Store"].nunique()}</div>
        <div class="kpi-label">Walmart Stores</div>
    </div>
    """)

with c3:
    st.html(f"""
    <div class="kpi">
        <div class="kpi-value">₹{df["Weekly_Sales"].mean():,.0f}</div>
        <div class="kpi-label">Average Weekly Sales</div>
    </div>
    """)

with c4:
    st.html("""
    <div class="kpi">
        <div class="kpi-value">2</div>
        <div class="kpi-label">Forecasting Models</div>
    </div>
    """)

# --------------------------------------------------
# PROJECT OVERVIEW
# --------------------------------------------------

st.html("""
<div class="section-title">Project Overview</div>
<div class="section-subtitle">
    Combining statistical forecasting and machine learning
    to understand retail sales behavior.
</div>
""")

cols = st.columns(4)
cards = [
    ("📊", "Data Analysis", "Historical Walmart sales analysis"),
    ("📈", "Forecasting", "Future sales prediction"),
    ("🤖", "Machine Learning", "XGBoost-based prediction"),
    ("⏳", "Time Series", "SARIMA-based forecasting")
]

for col, card in zip(cols, cards):
    with col:
        st.html(f"""
        <div class="card">
            <div class="card-icon">{card[0]}</div>
            <div class="card-title">{card[1]}</div>
            <div class="card-text">{card[2]}</div>
        </div>
        """)

# --------------------------------------------------
# BUSINESS PROBLEM
# --------------------------------------------------

st.html("""
<div class="section-title">Business Problem</div>
<div class="section-subtitle">
    Why accurate retail sales forecasting matters
</div>

<div class="card">
    <div class="card-text" style="font-size:16px; color:#334155;">
        Retail businesses need accurate sales forecasts to plan
        inventory, promotions, staffing and resources.
        This project analyzes historical Walmart sales data to
        identify sales patterns and support future demand prediction.
    </div>
</div>
""")

st.write("")

flow_cols = st.columns(4)
flow_items = [
    ("📂", "Historical Data"),
    ("🔍", "Analysis"),
    ("🔮", "Prediction"),
    ("💼", "Business Decision")
]

for col, item in zip(flow_cols, flow_items):
    with col:
        st.html(f"""
        <div class="flow">
            <div style="font-size:25px;">{item[0]}</div>
            {item[1]}
        </div>
        """)

# --------------------------------------------------
# DATASET
# --------------------------------------------------

st.html("""
<div class="section-title">Dataset</div>
<div class="section-subtitle">
    Overview of the Walmart retail sales data used in the project
</div>
""")

d1, d2 = st.columns(2)

with d1:
    st.html("""
    <div class="card">
        <div class="card-title">Dataset Information</div>
        <p class="card-text"><b>Dataset:</b> Walmart Sales</p>
        <p class="card-text"><b>Data Type:</b> Retail / Time Series</p>
        <p class="card-text"><b>Target:</b> Weekly Sales</p>
        <p class="card-text"><b>Granularity:</b> Store &amp; Department</p>
    </div>
    """)

with d2:
    st.html("""
    <div class="card">
        <div class="card-title">Key Factors</div>
        <p class="card-text">
            Date, Holiday indicators, Temperature,
            Fuel Price, CPI, Unemployment and other
            available sales-related features.
        </p>
    </div>
    """)
# --------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

st.html("""
<div class="section-title">🔍 Exploratory Data Analysis</div>
<div class="section-subtitle">
    Understanding sales trends, monthly patterns and holiday impact
</div>
""")

# ---------- 1. Overall Sales Trend ----------

st.html("""
<div class="card">
    <div class="card-title">📈 Overall Sales Trend</div>
    <div class="card-text">
        This graph shows how Walmart weekly sales changed over time.
        It helps identify overall trends and fluctuations in demand.
    </div>
</div>
""")

trend_df = (
    df.groupby("Date")["Weekly_Sales"]
    .sum()
    .reset_index()
    .sort_values("Date")
)

fig1, ax1 = plt.subplots(figsize=(12, 4.5))

ax1.plot(
    trend_df["Date"],
    trend_df["Weekly_Sales"],
    linewidth=2
)

ax1.set_title("Walmart Weekly Sales Trend")
ax1.set_xlabel("Date")
ax1.set_ylabel("Total Weekly Sales")
ax1.grid(alpha=0.2)

plt.tight_layout()
st.pyplot(fig1)

# ---------- 2. Monthly Pattern ----------

st.html("""
<div class="card">
    <div class="card-title">📅 Monthly Sales Pattern</div>
    <div class="card-text">
        Monthly aggregation helps identify recurring changes in
        sales throughout the year.
    </div>
</div>
""")

monthly_df = df.copy()
monthly_df["Month"] = monthly_df["Date"].dt.month

monthly_sales = (
    monthly_df.groupby("Month")["Weekly_Sales"]
    .mean()
    .reset_index()
)

fig2, ax2 = plt.subplots(figsize=(10, 4.5))

ax2.plot(
    monthly_sales["Month"],
    monthly_sales["Weekly_Sales"],
    marker="o",
    linewidth=2
)

ax2.set_title("Average Sales by Month")
ax2.set_xlabel("Month")
ax2.set_ylabel("Average Weekly Sales")
ax2.set_xticks(range(1, 13))
ax2.grid(alpha=0.2)

plt.tight_layout()
st.pyplot(fig2)

# ---------- 3. Holiday Impact ----------

st.html("""
<div class="card">
    <div class="card-title">🎉 Holiday Impact</div>
    <div class="card-text">
        This comparison shows the difference between average
        sales during holiday and non-holiday weeks.
    </div>
</div>
""")

holiday_sales = (
    df.groupby("IsHoliday")["Weekly_Sales"]
    .mean()
    .reset_index()
)

holiday_sales["Type"] = holiday_sales["IsHoliday"].map({
    False: "Non-Holiday",
    True: "Holiday"
})

fig3, ax3 = plt.subplots(figsize=(8, 4.5))

ax3.bar(
    holiday_sales["Type"],
    holiday_sales["Weekly_Sales"]
)

ax3.set_title("Average Sales: Holiday vs Non-Holiday")
ax3.set_xlabel("Week Type")
ax3.set_ylabel("Average Weekly Sales")
ax3.grid(axis="y", alpha=0.2)

plt.tight_layout()
st.pyplot(fig3)

# ---------- 4. Store Comparison ----------

st.html("""
<div class="card">
    <div class="card-title">🏪 Store Sales Comparison</div>
    <div class="card-text">
        This chart compares the average weekly sales across
        Walmart stores.
    </div>
</div>
""")

store_sales = (
    df.groupby("Store")["Weekly_Sales"]
    .mean()
    .sort_values(ascending=False)
)

fig4, ax4 = plt.subplots(figsize=(12, 5))

store_sales.plot(
    kind="bar",
    ax=ax4
)

ax4.set_title("Average Weekly Sales by Store")
ax4.set_xlabel("Store")
ax4.set_ylabel("Average Weekly Sales")
ax4.grid(axis="y", alpha=0.2)

plt.tight_layout()
st.pyplot(fig4)
# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

st.html("""
<div class="section-title">⚙️ Feature Engineering</div>
<div class="section-subtitle">
    Creating useful features for better sales forecasting
</div>
""")

st.html("""
<div class="card">
    <div class="card-title">📊 Forecasting Features</div>
    <div class="card-text">
        The forecasting system uses historical sales and time-based
        information to understand sales patterns and improve prediction.
    </div>
</div>
""")

feature_cols = [
    "Lag Features",
    "Rolling Mean",
    "Year",
    "Month",
    "Week of Year",
    "Quarter",
    "IsHoliday",
    "Temperature",
    "Fuel Price",
    "CPI",
    "Unemployment"
]

cols = st.columns(4)

for i, feature in enumerate(feature_cols):
    with cols[i % 4]:
        st.html(f"""
        <div class="card" style="text-align:center; min-height:100px;">
            <div class="card-title">{feature}</div>
        </div>
        """)
        # --------------------------------------------------
# MODELS
# --------------------------------------------------

st.html("""
<div class="section-title">🤖 Forecasting Models</div>
<div class="section-subtitle">
    Comparing statistical time series and machine learning approaches
</div>
""")

col1, col2 = st.columns(2)

with col1:
    st.html("""
    <div class="card" style="min-height:230px;">
        <div class="card-title">📈 SARIMA</div>
        <div class="card-text">
            <b>Statistical Time Series Model</b><br><br>
            • Handles time-based trends<br>
            • Captures seasonal patterns<br>
            • Uses historical sales behavior<br>
            • Suitable for time series forecasting
        </div>
    </div>
    """)

with col2:
    st.html("""
    <div class="card" style="min-height:230px;">
        <div class="card-title">🚀 XGBoost</div>
        <div class="card-text">
            <b>Machine Learning Model</b><br><br>
            • Uses engineered features<br>
            • Captures complex relationships<br>
            • Uses multiple influencing factors<br>
            • Suitable for predictive forecasting
        </div>
    </div>
    """)
    # --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.html("""
<div class="section-title">⚔️ Model Comparison</div>
<div class="section-subtitle">
    Comparing forecasting models based on prediction performance
</div>
""")

col1, col2 = st.columns(2)

with col1:
    st.html("""
    <div class="card" style="text-align:center; min-height:180px;">
        <div class="card-title">📈 SARIMA</div>
        <div class="card-text">
            Statistical time series forecasting model
        </div>
        <div style="font-size:28px; font-weight:800; margin-top:15px;">
            SARIMA
        </div>
    </div>
    """)

with col2:
    st.html("""
    <div class="card" style="text-align:center; min-height:180px;">
        <div class="card-title">🚀 XGBoost</div>
        <div class="card-text">
            Machine learning forecasting model
        </div>
        <div style="font-size:28px; font-weight:800; margin-top:15px;">
            XGBoost
        </div>
    </div>
    """)

st.html("""
<div class="card" style="margin-top:20px;">
    <div class="card-title">📊 Performance Metrics</div>
    <div class="card-text">
        RMSE and MAE will be used to compare forecasting accuracy.
        Lower error values indicate better predictive performance.
    </div>
</div>
""")
# --------------------------------------------------
# ACTUAL VS PREDICTED SALES
# --------------------------------------------------

st.html("""
<div class="section-title">📈 Actual vs Predicted Sales</div>
<div class="section-subtitle">
    Comparing actual historical sales with XGBoost predictions
</div>
""")

# Prepare evaluation data
eval_data = df.copy()

eval_data = eval_data.sort_values(
    ["Store", "Dept", "Date"]
).reset_index(drop=True)

# Create the same features used during XGBoost training
eval_data["Year"] = eval_data["Date"].dt.year
eval_data["Month"] = eval_data["Date"].dt.month
eval_data["Week"] = eval_data["Date"].dt.isocalendar().week.astype(int)
eval_data["Day"] = eval_data["Date"].dt.day
eval_data["Quarter"] = eval_data["Date"].dt.quarter
eval_data["IsHoliday"] = eval_data["IsHoliday"].astype(int)

eval_data["Lag_1"] = (
    eval_data.groupby(["Store", "Dept"])["Weekly_Sales"]
    .shift(1)
)

eval_data["Lag_2"] = (
    eval_data.groupby(["Store", "Dept"])["Weekly_Sales"]
    .shift(2)
)

eval_data["Lag_4"] = (
    eval_data.groupby(["Store", "Dept"])["Weekly_Sales"]
    .shift(4)
)

eval_data["Rolling_Mean_4"] = (
    eval_data.groupby(["Store", "Dept"])["Weekly_Sales"]
    .transform(lambda x: x.shift(1).rolling(4).mean())
)

# Remove rows where lag features are unavailable
eval_data = eval_data.dropna().reset_index(drop=True)

# Generate XGBoost predictions
eval_data["Predicted_Sales"] = xgb_model.predict(
    eval_data[xgb_features]
)

# Store selection
comparison_store = st.selectbox(
    "Select Store for Comparison",
    sorted(eval_data["Store"].unique()),
    key="comparison_store_select"
)

# Department selection
comparison_dept = st.selectbox(
    "Select Department for Comparison",
    sorted(
        eval_data[
            eval_data["Store"] == comparison_store
        ]["Dept"].unique()
    ),
    key="comparison_dept_select"
)

comparison_data = eval_data[
    (eval_data["Store"] == comparison_store) &
    (eval_data["Dept"] == comparison_dept)
].sort_values("Date").tail(20)

if len(comparison_data) > 0:

    fig_compare, ax_compare = plt.subplots(figsize=(12, 5))

    ax_compare.plot(
        comparison_data["Date"],
        comparison_data["Weekly_Sales"],
        marker="o",
        linewidth=2,
        label="Actual Sales"
    )

    ax_compare.plot(
        comparison_data["Date"],
        comparison_data["Predicted_Sales"],
        marker="o",
        linewidth=2,
        label="Predicted Sales"
    )

    ax_compare.set_title(
        "Actual vs Predicted Weekly Sales"
    )

    ax_compare.set_xlabel("Date")
    ax_compare.set_ylabel("Weekly Sales")

    ax_compare.legend()
    ax_compare.grid(alpha=0.2)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig_compare)

    st.info(
        "The graph compares actual Walmart sales with "
        "sales predicted by the XGBoost model."
    )
col1, col2, col3 = st.columns(3)

with col1:
    store = st.selectbox(
        "Select Store",
        sorted(df["Store"].dropna().unique()),
        key="forecast_store_select"
    )

with col2:
    department = st.selectbox(
        "Select Department",
        sorted(df["Dept"].dropna().unique()),
        key="forecast_department_select"
    )

with col3:
    forecast_period = st.selectbox(
        "Forecast Period",
        ["1 Week", "4 Weeks", "8 Weeks"],
        key="forecast_period_select"
    )

# --------------------------------------------------
# XGBOOST FORECAST GENERATION
# --------------------------------------------------

if st.button("🔮 Generate Forecast", use_container_width=True):

    filtered_data = df[
        (df["Store"] == store) &
        (df["Dept"] == department) 
    ].sort_values("Date").copy()

    if len(filtered_data) >= 4:

        # Number of weeks selected by user
        if forecast_period == "1 Week":
            weeks = 1
        elif forecast_period == "4 Weeks":
            weeks = 4
        else:
            weeks = 8

        # Keep historical data for recursive forecasting
        history = filtered_data.copy()

        forecasts = []

        for i in range(weeks):

            latest_date = history["Date"].iloc[-1]
            next_date = latest_date + pd.Timedelta(weeks=1)

            # Create features for next week
            prediction_data = pd.DataFrame([{
                "Store": store,
                "Dept": department,
                "Year": next_date.year,
                "Month": next_date.month,
                "Week": int(next_date.isocalendar().week),
                "Day": next_date.day,
                "Quarter": next_date.quarter,
                "IsHoliday": 0,
                "Lag_1": history["Weekly_Sales"].iloc[-1],
                "Lag_2": history["Weekly_Sales"].iloc[-2],
                "Lag_4": history["Weekly_Sales"].iloc[-4],
                "Rolling_Mean_4": history["Weekly_Sales"].tail(4).mean()
            }])

            # Keep exact training feature order
            prediction_data = prediction_data[xgb_features]

            # XGBoost prediction
            predicted_sales = xgb_model.predict(prediction_data)[0]

            # Save forecast
            forecasts.append({
                "Date": next_date,
                "Predicted Sales": predicted_sales
            })

            # Add prediction to history for next recursive forecast
            new_row = history.iloc[-1].copy()
            new_row["Date"] = next_date
            new_row["Weekly_Sales"] = predicted_sales

            history = pd.concat(
                [history, pd.DataFrame([new_row])],
                ignore_index=True
            )

        # Convert forecast results to DataFrame
        forecast_df = pd.DataFrame(forecasts)

        # --------------------------------------------------
        # FORECAST RESULT
        # --------------------------------------------------

        total_forecast = forecast_df["Predicted Sales"].sum()
        average_forecast = forecast_df["Predicted Sales"].mean()

        st.html(f"""
        <div class="card" style="text-align:center; margin-top:25px;">

            <div class="card-title">
                🔮 XGBoost Forecast Result
            </div>

            <div style="font-size:36px; font-weight:800; margin:15px 0;">
                ₹{total_forecast:,.0f}
            </div>

            <div class="card-text">
                Store: {store}<br>
                Department: {department}<br>
                Forecast Period: {forecast_period}<br>
                Average Weekly Forecast: ₹{average_forecast:,.0f}<br>
                <b>Model: XGBoost</b>
            </div>

        </div>
        """)

        st.success(
            f"{weeks}-week forecast generated successfully using XGBoost."
        )

        # --------------------------------------------------
        # FORECAST GRAPH
        # --------------------------------------------------

        st.markdown("### 📈 Future Sales Forecast")

        fig_forecast, ax_forecast = plt.subplots(figsize=(12, 5))

        ax_forecast.plot(
            forecast_df["Date"],
            forecast_df["Predicted Sales"],
            marker="o",
            linewidth=2
        )

        ax_forecast.set_title(
            f"XGBoost Sales Forecast — {forecast_period}"
        )

        ax_forecast.set_xlabel("Forecast Date")
        ax_forecast.set_ylabel("Predicted Sales")
        ax_forecast.grid(alpha=0.2)

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig_forecast)

        # --------------------------------------------------
        # FORECAST TABLE
        # --------------------------------------------------

        st.markdown("### 📋 Forecast Details")

        display_forecast = forecast_df.copy()

        display_forecast["Date"] = display_forecast["Date"].dt.strftime(
            "%d-%m-%Y"
        )

        display_forecast["Predicted Sales"] = (
            display_forecast["Predicted Sales"]
            .map(lambda x: f"₹{x:,.0f}")
        )

        st.dataframe(
            display_forecast,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Not enough sales history available for this Store and Department."
        )

# --------------------------------------------------
# HISTORICAL SALES PREVIEW
# --------------------------------------------------

st.html("""
<div class="section-title">Sales Trend Preview</div>
<div class="section-subtitle">
    Historical weekly sales for the selected Walmart store
</div>
""")

selected_store = st.selectbox(
    "Select Store",
    sorted(df["Store"].unique()),
    key="historical_store_select"
)

store_df = df[df["Store"] == selected_store].sort_values("Date")

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(
    store_df["Date"],
    store_df["Weekly_Sales"],
    linewidth=2
)
ax.set_xlabel("Date")
ax.set_ylabel("Weekly Sales")
ax.grid(alpha=0.2)
plt.tight_layout()
st.pyplot(fig)



# --------------------------------------------------
# PROJECT RESULTS
# --------------------------------------------------

st.html("""
<div class="section-title">Project Results</div>
<div class="section-subtitle">
    Main outcomes of the forecasting project
</div>
""")

result_cols = st.columns(4)
results = [
    ("📊", "Historical Patterns", "Sales trends and patterns identified"),
    ("🔍", "Important Factors", "Key sales-related variables analyzed"),
    ("⚔️", "Model Comparison", "SARIMA and XGBoost included"),
    ("🔮", "Forecasting", "Future sales prediction dashboard")
]

for col, item in zip(result_cols, results):
    with col:
        st.html(f"""
        <div class="card">
            <div class="card-icon">{item[0]}</div>
            <div class="card-title">{item[1]}</div>
            <div class="card-text">{item[2]}</div>
        </div>
        """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.html("""
<div class="footer">
    <div class="footer-title">
        Walmart Retail Sales Forecasting
    </div>

    <div style="margin-top:8px;">
        Time Series Analysis • Machine Learning • Business Analytics
    </div>

    <div style="margin-top:15px; font-size:13px;">
        Academic / Internship Project
    </div>
</div>
""")
