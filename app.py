import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🛒",
    layout="wide"
)

@st.cache_resource
def load_model():
    model         = joblib.load('models/churn_model.joblib')
    scaler        = joblib.load('models/scaler.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# Header
st.markdown("<h1 style='text-align:center'>🛒 Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray'>E-Commerce | Random Forest | ROC-AUC: 0.9976</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.markdown("## 📋 Customer Profile")
tenure             = st.sidebar.slider("Tenure (months)", 0, 60, 12)
satisfaction_score = st.sidebar.slider("Satisfaction Score (1-5)", 1, 5, 3)
complain           = st.sidebar.selectbox("Has Complained?", [0, 1], format_func=lambda x: "✅ No" if x==0 else "⚠️ Yes")
cashback_amount    = st.sidebar.slider("Cashback Amount ($)", 0, 300, 150)
order_count        = st.sidebar.slider("Order Count", 1, 16, 4)
days_since_last    = st.sidebar.slider("Days Since Last Order", 0, 30, 5)
city_tier          = st.sidebar.radio("City Tier", [1, 2, 3], horizontal=True)
num_devices        = st.sidebar.slider("Devices Registered", 1, 6, 2)
payment_mode       = st.sidebar.selectbox("Payment Mode",
                        ["Debit Card","UPI","Credit Card","Cash on Delivery","E wallet"])
marital_status     = st.sidebar.selectbox("Marital Status",
                        ["Single","Married","Divorced"])

predict_btn = st.sidebar.button("Predict", type="primary", use_container_width=True)

def make_prediction():
    row = {col: 0.0 for col in feature_names}
    row['Tenure']                   = float(tenure)
    row['SatisfactionScore']        = float(satisfaction_score)
    row['Complain']                 = float(complain)
    row['CashbackAmount']           = float(cashback_amount)
    row['OrderCount']               = float(order_count)
    row['DaySinceLastOrder']        = float(days_since_last)
    row['CityTier']                 = float(city_tier)
    row['NumberOfDeviceRegistered'] = float(num_devices)
    row['ARPU_proxy']               = cashback_amount / (order_count + 1)
    row['contract_risk']            = 1.0 if (complain==1 and satisfaction_score<=2) else 0.0
    row['recency_score']            = float(1 / (days_since_last + 1))
    row['device_loyalty']           = float(num_devices / (tenure + 1))

    payment_map = {
        'UPI':              'PreferredPaymentMode_UPI',
        'Credit Card':      'PreferredPaymentMode_Credit Card',
        'Cash on Delivery': 'PreferredPaymentMode_Cash on Delivery',
        'E wallet':         'PreferredPaymentMode_E wallet',
    }
    if payment_mode in payment_map and payment_map[payment_mode] in row:
        row[payment_map[payment_mode]] = 1.0

    ms_map = {'Married': 'MaritalStatus_Married', 'Single': 'MaritalStatus_Single'}
    if marital_status in ms_map and ms_map[marital_status] in row:
        row[ms_map[marital_status]] = 1.0

    X_input  = pd.DataFrame([row])[feature_names]
    X_scaled = pd.DataFrame(scaler.transform(X_input), columns=feature_names)
    prob     = float(model.predict_proba(X_scaled)[0, 1])
    return prob

if predict_btn:
    prob = make_prediction()

    # Risk level
    if prob >= 0.65:
        risk, color, emoji = "HIGH", "red", "🔴"
    elif prob >= 0.35:
        risk, color, emoji = "MEDIUM", "orange", "🟡"
    else:
        risk, color, emoji = "LOW", "green", "🟢"

    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric(" Churn Probability", f"{prob*100:.1f}%")
    col2.metric(" Retention Probability", f"{(1-prob)*100:.1f}%")
    col3.metric("⚠️ Risk Level", f"{emoji} {risk}")

    st.markdown("---")

    # Risk message
    if risk == "HIGH":
        st.error(f"""
        ### 🔴 High Churn Risk Detected!
        **Recommended Actions:**
        - 📞 Contact customer within 24 hours
        - 🎁 Offer personalized discount or loyalty bonus
        - 🔧 Resolve all open complaints immediately
        - 💬 Schedule a customer satisfaction call
        """)
    elif risk == "MEDIUM":
        st.warning(f"""
        ### 🟡 Medium Churn Risk — Monitor Closely
        **Recommended Actions:**
        - 📧 Send re-engagement email with coupon
        - 💰 Offer cashback bonus on next order
        - 📊 Monitor purchase activity for 7 days
        """)
    else:
        st.success(f"""
        ### 🟢 Low Risk — Customer is Stable
        **Recommended Actions:**
        - ⭐ Encourage product reviews
        - 🎯 Maintain loyalty through cashback program
        - 📱 Promote new app features
        """)

    st.markdown("---")

    # Profile summary
    st.subheader(" Customer Profile")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **Tenure:** {tenure} months")
        st.markdown(f"- **Satisfaction Score:** {satisfaction_score}/5")
        st.markdown(f"- **Complained:** {'Yes ⚠️' if complain else 'No ✅'}")
        st.markdown(f"- **Cashback Amount:** ${cashback_amount}")
        st.markdown(f"- **Order Count:** {order_count}")
    with col2:
        st.markdown(f"- **Days Since Last Order:** {days_since_last}")
        st.markdown(f"- **City Tier:** {city_tier}")
        st.markdown(f"- **Devices Registered:** {num_devices}")
        st.markdown(f"- **Payment Mode:** {payment_mode}")
        st.markdown(f"- **Marital Status:** {marital_status}")

else:
    # Welcome screen
    st.markdown("### 👈 Fill in the customer profile and click **Predict**")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("- Model Accuracy", "97.96%")
    col2.metric("- ROC-AUC Score", "0.9976")
    col3.metric("- Training Samples", "7,492")

    st.markdown("---")
    st.subheader("📈 Top Churn Drivers")
    data = {
        'Feature':    ['Tenure', 'Complain', 'DaySinceLastOrder', 'SatisfactionScore', 'CashbackAmount'],
        'Effect':     ['↓ Churn', '↑ Churn', '↑ Churn', '↑ Churn', '↓ Churn'],
        'Insight':    [
            'Longer tenure = more loyal customers',
            'Complaints strongly predict churn',
            'Inactive customers likely to leave',
            'Low satisfaction drives churn',
            'Higher cashback = better retention'
        ]
    }
    st.table(pd.DataFrame(data))
