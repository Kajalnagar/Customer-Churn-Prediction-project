import streamlit as st
import pandas as pd
import plotly.express as px
import joblib


# PAGE CONFIG


st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)


# LOAD DATA


df = pd.read_csv("customer_churn.csv")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# SIDEBAR


st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Analytics",
        "Prediction",
        "Recommendations"
    ]
)


# DASHBOARD PAGE


if page == "Dashboard":

    st.title("📈 Customer Churn Dashboard")

    total_customers = len(df)

    churn_rate = round(
        (df["Churn"].sum()/len(df))*100,
        2
    )

    avg_tenure = round(
        df["Tenure"].mean(),
        2
    )

    avg_charges = round(
        df["MonthlyCharges"].mean(),
        2
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Customers",
        total_customers
    )

    c2.metric(
        "Churn Rate %",
        churn_rate
    )

    c3.metric(
        "Average Tenure",
        avg_tenure
    )

    c4.metric(
        "Avg Monthly Charges",
        avg_charges
    )

    st.markdown("---")

    fig1 = px.pie(
        df,
        names="Churn",
        title="Customer Churn Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )


# ANALYTICS PAGE


elif page == "Analytics":

    st.title("📊 Customer Analytics")

    fig2 = px.histogram(
        df,
        x="Contract",
        color="Churn",
        barmode="group",
        title="Contract vs Churn"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    fig3 = px.histogram(
        df,
        x="PaymentMethod",
        color="Churn",
        title="Payment Method Analysis"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    fig4 = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        title="Monthly Charges vs Churn"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    fig5 = px.box(
        df,
        x="Churn",
        y="TotalCharges",
        title="Total Charges vs Churn"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )


# PREDICTION PAGE


elif page == "Prediction":

    st.title("🤖 Churn Prediction")

    tenure = st.slider(
        "Tenure",
        0,
        72,
        12
    )

    monthly = st.number_input(
        "Monthly Charges",
        value=50.0
    )

    total = st.number_input(
        "Total Charges",
        value=1000.0
    )

    contract = st.selectbox(
        "Contract",
        [0,1,2]
    )

    payment = st.selectbox(
        "Payment Method",
        [0,1,2]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        [0,1]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0,1]
    )

    if st.button("Predict Churn"):

        sample = pd.DataFrame({

            "Tenure":[tenure],
            "MonthlyCharges":[monthly],
            "TotalCharges":[total],
            "Contract":[contract],
            "PaymentMethod":[payment],
            "PaperlessBilling":[paperless],
            "SeniorCitizen":[senior]

        })

        sample_scaled = scaler.transform(sample)

        prediction = model.predict(
            sample_scaled
        )

        probability = model.predict_proba(
            sample_scaled
        )[0][1]

        st.markdown("---")

        if prediction[0] == 1:

            st.error(
                "⚠ Customer Likely To Churn"
            )

        else:

            st.success(
                "✅ Customer Likely To Stay"
            )

        st.write(
            f"Churn Probability: {round(probability*100,2)}%"
        )

# RECOMMENDATIONS PAGE


elif page == "Recommendations":

    st.title("💡 Business Recommendations")

    st.success("""
    1. Offer loyalty rewards to long-term customers.

    2. Provide discounts to customers with high churn risk.

    3. Encourage annual contracts instead of monthly contracts.

    4. Improve customer support quality.

    5. Promote auto-payment methods.

    6. Create retention campaigns for senior citizens.

    7. Target high-risk customers using model predictions.
    """)

    st.info("""
    These recommendations are generated based on churn analysis and machine learning results.
    """)
