import streamlit as st
import json

# Function to save ARIMA parameters to a JSON file
def save_arima_parameters(p, d, q):
    params = {'p': int(p), 'd': int(d), 'q': int(q)}
    with open('arima_config.json', 'w') as config_file:
        json.dump(params, config_file)
    st.success(f"Parameters saved: p={p}, d={d}, q={q}")

# Function to reset ARIMA parameters to default
def reset_arima_parameters():
    save_arima_parameters(5, 1, 0)
    st.success("Parameters reset to default values: p=5, d=1, q=0")

# Function to handle login
def login():
    username = st.text_input("Username", key="username")
    password = st.text_input("Password", type="password", key="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state['logged_in'] = True
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

# Main function that runs the Streamlit app
def admin_panel():
    st.title("Admin Panel")
    
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        login()
    else:
        

        st.subheader("Set ARIMA Parameters")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("Enter value for p (AR term)", min_value=0, value=5, step=1, key='p')
        with col2:
            d = st.number_input("Enter value for d (Differencing order)", min_value=0, value=1, step=1, key='d')
        with col3:
            q = st.number_input("Enter value for q (MA term)", min_value=0, value=0, step=1, key='q')

        if st.button("Save Parameters"):
            save_arima_parameters(p, d, q)
        
        if st.button("Reset Parameters to Default"):
            reset_arima_parameters()

        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.experimental_rerun()


