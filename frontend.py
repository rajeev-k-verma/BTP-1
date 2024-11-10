import streamlit as st
import requests
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="ML Model Frontend",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title
st.title('Material-structure- processing- performance correlations using AI/ML for polymeric liner material in Type-IV hydrogen pressure vessel 🤖')

# Project name and description
st.markdown("""
    #### Objective:
    The goal is to predict the permeability and permeation rate, and to find the optimal material that minimizes the material cost while maintaining structural integrity.
""")

# Layout the page into two columns
col1, col2 = st.columns(2)

# Input section
with col1:
    st.markdown("<h2 style='font-size:24px;'>Select Prediction Type</h2>", unsafe_allow_html=True)
    prediction_type = st.radio('', ['Predict with M_NAME', 'Find Optimal M_NAME'], index=1)

    st.subheader('Input Features')
    input_data = {
        'C': st.number_input('Capacity (C)', value=1),
        'LD': st.number_input('Length to Diameter Ratio (LD)', value=2),
        'DI': st.number_input('Internal Diameter (DI)', value=91.3913),
        'TL': st.number_input('Thickness (TL)', value=5),
        'L_CYL': st.number_input('Length of Cylinder (L_CYL)', value=29110.8648),
        'PO': st.number_input('Operating Pressure (PO)', value=201)
    }

    # Fetch backend URL from environment variable (this should be set on Streamlit Cloud)
    BACKEND_URL = st.secrets["BACKEND_URL"]

    if prediction_type == 'Find Optimal M_NAME':
        endpoint = f'{BACKEND_URL}/find_optimal_mname'
    else:
        input_data['M_NAME'] = st.selectbox('Material Name (M_NAME)', ['PA1B', 'PA2R', 'HD2R', 'HD1B', 'HD1C'])
        endpoint = f'{BACKEND_URL}/predict_with_mname'

# Output section
with col2:
    st.subheader('Predictions')
    if st.button('Get Predictions'):
        with st.spinner('Getting predictions...'):
            response = requests.post(endpoint, json=input_data)
            if response.status_code == 200:
                predictions = response.json()
                st.success("Predictions received successfully!")
                st.write("### Predictions")
                st.write(f"**Material Name (M_NAME)**: {predictions.get('M_NAME', 'N/A')}")
                st.write(f"**Material Cost (MC)**: {predictions.get('MC', 'N/A')}")
                st.write(f"**Permeation Rate (PR_NCC)**: {predictions.get('PR_NCC', 'N/A')}")
                st.write(f"**Permeability (PER_FIT)**: {predictions.get('PER_FIT', 'N/A')}")
                st.balloons()
            else:
                st.error("Error: Unable to get predictions. Please try again.")

# Button to display pandas profiling report in the main area
if st.button('Show Training Data Profiling Report'):
    with st.spinner('Loading profiling report...'):
        profiling_html_url = 'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/ydata_profiling_report.html'
        profiling_html = requests.get(profiling_html_url).text
        st.components.v1.html(profiling_html, height=800, scrolling=True)

# Button to show training graphs in the main area
if st.button('Show Training Graphs'):
    graph_urls = [
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Actual%20vs%20Predicted%20for%20MC.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Actual%20vs%20Predicted%20for%20PER_FIT.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Actual%20vs%20Predicted%20for%20PR_NCC.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Distribution%20of%20residuals%20for%20MC.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Distribution%20of%20residuals%20for%20PER_FIT.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Distribution%20of%20residuals%20for%20PR_NCC.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Feature%20importance%20for%20MC.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Feature%20importance%20for%20PER_FIT.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Feature%20importance%20for%20PR_NCC.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Learning%20curve%20for%20MC.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Learning%20curve%20for%20PER_FIT.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Learning%20curve%20for%20PR_NCC.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Residual%20plot%20for%20MC.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Residual%20plot%20for%20PER_FIT.png',
        'https://raw.githubusercontent.com/rajeev-k-verma/BTP-1/refs/heads/main/reports/Residual%20plot%20for%20PR_NCC.png'
    ]
    for url in graph_urls:
        st.image(url, use_column_width=True)

# Footer
st.markdown("""
    ---
    <div style='text-align: center;'>
        <p>Created with <span style="color: #FF5733;">💡</span> by <strong>Rajeev Kumar Verma</strong> from <span style="color: #FFC300;">🇮🇳</span></p>
        <p>Connect with me on <a href='https://github.com/rajeev-k-verma' target='_blank'>GitHub</a> |
        <a href='https://www.linkedin.com/in/rajeev-k-verma' target='_blank'>LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)
