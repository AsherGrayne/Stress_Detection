import streamlit as st
import time
from database import create_user, authenticate_user, init_database

def show_signup_page():
    """Display the sign up page with professional minimalist design"""
    # Center the form with custom CSS
    st.markdown("""
        <style>
        .auth-container {
            max-width: 420px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        .auth-box {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 40px 32px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .auth-title {
            font-size: 28px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 8px;
            color: #fafafa;
        }
        .auth-subtitle {
            font-size: 14px;
            text-align: center;
            color: #b0b0b0;
            margin-bottom: 32px;
        }
        .divider {
            display: flex;
            align-items: center;
            text-align: center;
            margin: 24px 0;
            color: #666;
            font-size: 12px;
        }
        .divider::before,
        .divider::after {
            content: '';
            flex: 1;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .divider::before {
            margin-right: 10px;
        }
        .divider::after {
            margin-left: 10px;
        }
        .social-button {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background-color: rgba(255, 255, 255, 0.05);
            color: #fafafa;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .social-button:hover {
            background-color: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }
        .switch-auth {
            text-align: center;
            margin-top: 24px;
            font-size: 14px;
            color: #b0b0b0;
        }
        .switch-auth-link {
            color: #1f77b4;
            text-decoration: none;
            font-weight: 500;
            cursor: pointer;
        }
        .switch-auth-link:hover {
            text-decoration: underline;
        }
        /* Style Streamlit buttons to match minimalist design */
        div[data-testid="stButton"] > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #fafafa;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background-color: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }
        div[data-testid="stForm"] {
            margin-top: 0;
        }
        div[data-testid="stTextInput"] > div > div > input {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #fafafa;
            border-radius: 8px;
        }
        div[data-testid="stTextInput"] > div > div > input:focus {
            border-color: #1f77b4;
            box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Container
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)
        
        # Title
        st.markdown('<div class="auth-title">Create Account</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Sign up to track your stress levels</div>', unsafe_allow_html=True)
        
        # Social sign-in buttons
        if st.button("🔵 Continue with Google", use_container_width=True, key="google_signup"):
            st.info("Google sign-in integration requires OAuth setup. Using email sign-up for now.")
        
        if st.button("⚫ Continue with Apple", use_container_width=True, key="apple_signup"):
            st.info("Apple sign-in integration requires OAuth setup. Using email sign-up for now.")
        
        st.markdown('<div class="divider">or</div>', unsafe_allow_html=True)
        
        # Sign up form
        with st.form("signup_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username", key="signup_username")
            email = st.text_input("Email", placeholder="Enter your email", key="signup_email")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="signup_confirm")
            
            submitted = st.form_submit_button("Sign Up", use_container_width=True, type="primary")
            
            if submitted:
                if not username or not email or not password:
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    success, error_message = create_user(username, email, password)
                    if success:
                        st.success("Account created successfully! Please sign in.")
                        st.session_state.show_signup = False
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Error: {error_message}")
        
        # Switch to sign in
        st.markdown(
            '<div class="switch-auth">Already have an account? <span class="switch-auth-link" onclick="window.location.reload()">Sign In</span></div>',
            unsafe_allow_html=True
        )
        if st.button("Sign In", use_container_width=True, key="switch_to_signin"):
            st.session_state.show_signup = False
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def show_signin_page():
    """Display the sign in page with professional minimalist design"""
    # Center the form with custom CSS
    st.markdown("""
        <style>
        .auth-container {
            max-width: 420px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        .auth-box {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 40px 32px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .auth-title {
            font-size: 28px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 8px;
            color: #fafafa;
        }
        .auth-subtitle {
            font-size: 14px;
            text-align: center;
            color: #b0b0b0;
            margin-bottom: 32px;
        }
        .divider {
            display: flex;
            align-items: center;
            text-align: center;
            margin: 24px 0;
            color: #666;
            font-size: 12px;
        }
        .divider::before,
        .divider::after {
            content: '';
            flex: 1;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .divider::before {
            margin-right: 10px;
        }
        .divider::after {
            margin-left: 10px;
        }
        .social-button {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background-color: rgba(255, 255, 255, 0.05);
            color: #fafafa;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .social-button:hover {
            background-color: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }
        .switch-auth {
            text-align: center;
            margin-top: 24px;
            font-size: 14px;
            color: #b0b0b0;
        }
        .switch-auth-link {
            color: #1f77b4;
            text-decoration: none;
            font-weight: 500;
            cursor: pointer;
        }
        .switch-auth-link:hover {
            text-decoration: underline;
        }
        /* Style Streamlit buttons to match minimalist design */
        div[data-testid="stButton"] > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #fafafa;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background-color: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }
        div[data-testid="stForm"] {
            margin-top: 0;
        }
        div[data-testid="stTextInput"] > div > div > input {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #fafafa;
            border-radius: 8px;
        }
        div[data-testid="stTextInput"] > div > div > input:focus {
            border-color: #1f77b4;
            box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Container
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)
        
        # Title
        st.markdown('<div class="auth-title">Welcome Back</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Sign in to access your dashboard</div>', unsafe_allow_html=True)
        
        # Social sign-in buttons
        if st.button("🔵 Continue with Google", use_container_width=True, key="google_signin"):
            st.info("Google sign-in integration requires OAuth setup. Using email sign-in for now.")
        
        if st.button("⚫ Continue with Apple", use_container_width=True, key="apple_signin"):
            st.info("Apple sign-in integration requires OAuth setup. Using email sign-in for now.")
        
        st.markdown('<div class="divider">or</div>', unsafe_allow_html=True)
        
        # Sign in form
        with st.form("signin_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username", key="signin_username")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="signin_password")
            
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")
            
            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.success(f"Welcome back, {user['username']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        
        # Switch to sign up
        st.markdown(
            '<div class="switch-auth">Don\'t have an account? <span class="switch-auth-link">Sign Up</span></div>',
            unsafe_allow_html=True
        )
        if st.button("Sign Up", use_container_width=True, key="switch_to_signup"):
            st.session_state.show_signup = True
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def check_authentication():
    """Check if user is authenticated, show auth pages if not"""
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    
    # If not authenticated, show auth pages
    if not st.session_state.authenticated:
        if st.session_state.show_signup:
            show_signup_page()
        else:
            show_signin_page()
        return False
    
    return True

def logout():
    """Logout the current user"""
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()

