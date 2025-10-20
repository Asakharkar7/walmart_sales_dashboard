import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
from prophet import Prophet
import subprocess
import json

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_parquet(Path("artifacts") / "walmart_merged.parquet")
    return df

df = load_data()

# -----------------------------------------------------
# Sidebar Filters
# -----------------------------------------------------
st.sidebar.header("🔍 Filters")
store = st.sidebar.selectbox("Select Store", sorted(df["Store"].unique()))
dept = st.sidebar.selectbox("Select Department", sorted(df["Dept"].unique()))
year = st.sidebar.selectbox("Select Year", sorted(df["Year"].unique()))

filtered = df[(df["Store"] == store) & (df["Dept"] == dept) & (df["Year"] == year)]

# -----------------------------------------------------
# Main Layout
# -----------------------------------------------------
st.title("🏪 Walmart Sales Forecasting Dashboard")
st.markdown("Explore trends, patterns, and AI-powered insights for Walmart sales data.")

st.divider()

# -----------------------------------------------------
# Tabs Section
# -----------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔮 Forecasting", "💬 AI Insights"])

# -----------------------------------------------------
# 📊 Tab 1: Overview
# -----------------------------------------------------
with tab1:
    st.subheader("📈 Weekly Sales Trend")
    fig = px.line(
        filtered,
        x="Date",
        y="Weekly_Sales",
        title="Weekly Sales Over Time",
        labels={"Weekly_Sales": "Weekly Sales", "Date": "Date"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write("#### Summary Statistics")
    st.dataframe(filtered.describe())

# -----------------------------------------------------
# 🔮 Tab 2: Forecasting
# -----------------------------------------------------
with tab2:
    st.subheader("📅 Forecast Future Sales")

    df_forecast = filtered[["Date", "Weekly_Sales"]].rename(columns={"Date": "ds", "Weekly_Sales": "y"})
    periods = st.slider("Select number of weeks to forecast:", 4, 52, 12)

    if st.button("Generate Forecast"):
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df_forecast)
        future = model.make_future_dataframe(periods=periods, freq="W")
        forecast = model.predict(future)

        fig1 = px.line(
            forecast,
            x="ds",
            y="yhat",
            title="Forecasted Weekly Sales",
            labels={"ds": "Date", "yhat": "Predicted Sales"},
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.write("### Forecast Data")
        st.dataframe(forecast.tail(periods))

# -----------------------------------------------------
# 💬 Tab 3: AI Insights (Restored Stable + Header Fix + Bold Fix)
# -----------------------------------------------------
import re
import subprocess

with tab3:
    st.subheader("🧠 AI-Generated Insights")

    if st.button("Generate AI Insight"):
        avg_sales = filtered["Weekly_Sales"].mean()
        max_sales = filtered["Weekly_Sales"].max()
        min_sales = filtered["Weekly_Sales"].min()

        avg_sales_str = f"${avg_sales:,.2f}"
        max_sales_str = f"${max_sales:,.2f}"
        min_sales_str = f"${min_sales:,.2f}"

        prompt = f"""
You are a senior retail analyst reviewing Walmart weekly sales data.

Here are the key statistics for **Store {store}, Department {dept}, Year {year}**:
- **Average Weekly Sales:** {avg_sales_str}
- **Maximum Weekly Sales:** {max_sales_str}
- **Minimum Weekly Sales:** {min_sales_str}

Please:
1. Write a short analytical summary describing trends and sales patterns.
2. Include clearly separated sections titled **Key Trends** and **Actionable Insights**.
3. Use Markdown formatting (lists, bullet points) and retain dollar signs for currency.
4. Avoid repeating input values exactly; paraphrase naturally.
"""

        try:
            with st.spinner("🤖 Generating insights with Ollama..."):
                process = subprocess.Popen(
                    ["ollama", "run", "mistral"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8"
                )

                output, error = process.communicate(input=prompt, timeout=300)

                clean_output = output or ""
                clean_output = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", clean_output)
                clean_output = re.sub(r"(\#{1,3})([A-Za-z])", r"\1 \2", clean_output)  # fix header jam
                clean_output = re.sub(r"#(\s|$)", "", clean_output)                  # remove lone #
                clean_output = clean_output.strip()

                if process.returncode == 0 and clean_output:
                    st.markdown("### 💡 Insight Generated")

                    formatted_output = clean_output
                    formatted_output = formatted_output.replace("### ", "\n\n### ")
                    formatted_output = formatted_output.replace("## ", "\n\n## ")
                    formatted_output = formatted_output.replace("- ", "\n- ")
                    formatted_output = re.sub(r"(\d)\.", r"\1. ", formatted_output)
                    formatted_output = re.sub(r"\n{3,}", "\n\n", formatted_output)
                    formatted_output = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", formatted_output)

                    st.markdown(
                        f"""
                        <div style='background-color:#0d3b29;
                                    padding:18px 20px;
                                    border-radius:12px;
                                    margin-top:10px;'>
                            <div style='color:#e6ffe6;
                                        font-size:16px;
                                        line-height:1.45;
                                        white-space:pre-wrap;'>
                                {formatted_output}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.error("⚠️ Ollama didn't produce a valid response.")

        except subprocess.TimeoutExpired:
            st.error("⏰ Ollama took too long (timed out after 300s). Try again.")
        except FileNotFoundError:
            st.error("⚠️ Ollama not found. Please ensure it's installed and available in PATH.")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {str(e)}")

