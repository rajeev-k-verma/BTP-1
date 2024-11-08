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
st.title('ML Model Frontend 🤖')

# Project name and description
st.markdown("""
    ### Material-structure- processing- performance correlations using AI/ML for polymeric liner material in Type-IV hydrogen pressure vessel
    This application allows you to interact with a machine learning model to address the challenge of hydrogen permeation in type 4 liners used for hydrogen storage.

    #### Objective:
    The goal is to predict the permeability and permeation rate, and to find the optimal material that minimizes the material cost while maintaining structural integrity.

    #### Input Parameters:
    The features that influence the outputs are as follows:
    1. **C**: Capacity of the cylinder (L)
    2. **LD**: Length to diameter ratio (dimensionless)
    3. **DI**: Internal diameter of the cylinder (mm)
    4. **TL**: Thickness of the cylinder (mm)
    5. **SA_D**: Surface area of the dome section (mm²)
    6. **SA_CYL**: Surface area of the cylinder section (mm²)
    7. **PO**: Operating pressure (bar)
    8. **M_NAME**: Name of the liner material (possibly encoded if categorical)

    #### Output Parameters:
    The target variables you want to predict are:
    1. **PER_FIT**: Permeability in mol H₂/m²/s/Pa
    2. **PR_NCC**: Permeation rate in Ncc/h/L
    3. **MC**: Total material cost (INR)
""")

# Option to select prediction type
st.subheader('Select Prediction Type')
prediction_type = st.radio('', ['Predict with M_NAME', 'Find Optimal M_NAME'])

st.subheader('Input Features')
input_data = {
    'C': st.number_input('C', value=1),
    'LD': st.number_input('LD', value=2),
    'DI': st.number_input('DI', value=91.3913),
    'TL': st.number_input('TL', value=5),
    'SA_D': st.number_input('SA_D', value=32296.1653),
    'SA_CYL': st.number_input('SA_CYL', value=29110.8648),
    'PO': st.number_input('PO', value=201)
}

# Fetch backend URL from environment variable (this should be set on Streamlit Cloud)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

if prediction_type == 'Predict with M_NAME':
    input_data['M_NAME'] = st.selectbox('M_NAME', ['K-X09086', 'FLX40HP', 'HD614', 'B5500'])
    endpoint = f'{BACKEND_URL}/predict_with_mname'
else:
    endpoint = f'{BACKEND_URL}/find_optimal_mname'

# Button to trigger prediction
if st.button('Predict'):
    with st.spinner('Getting predictions...'):
        response = requests.post(endpoint, json=input_data)
        if response.status_code == 200:
            predictions = response.json()
            st.success("Predictions received successfully!")
            st.markdown(f"""
                ### Predictions
                **M_NAME**: {predictions['M_NAME']}
                **MC**: {predictions['MC']}
                **PR_NCC**: {predictions['PR_NCC']}
                **PER_FIT**: {predictions['PER_FIT']}
            """)
            st.balloons()
        else:
            st.error("Error: Unable to get predictions. Please try again.")

# Button to display pandas profiling report in the main area
if st.button('Show Profiling Report'):
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
        <p>Developed by <strong>Rajeev Kumar Verma</strong> with 💖 from 🇮🇳</p>
        <p>Follow me on <a href='https://github.com/rajeev-k-verma' target='_blank'>GitHub</a> |
        <a href='https://www.linkedin.com/in/rajeev-verma' target='_blank'>LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)
