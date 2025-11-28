import streamlit as st
import time
from database import create_user, authenticate_user, init_database

def apply_auth_styles():
    """Apply custom CSS styles for authentication pages"""
    st.markdown("""
    <style>
    .auth-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
        padding: 2rem 0;
    }
    .auth-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        max-width: 450px;
        width: 100%;
        color: white;
    }
    .auth-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .auth-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    .auth-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .auth-form {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 1.5rem;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .auth-switch {
        text-align: center;
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }
    .auth-switch p {
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 0.5rem;
    }
    .auth-switch button {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .auth-switch button:hover {
        background: rgba(255, 255, 255, 0.3);
        border-color: rgba(255, 255, 255, 0.5);
    }
    .icon-wrapper {
        text-align: center;
        margin-bottom: 1rem;
    }
    .icon-wrapper svg {
        width: 60px;
        height: 60px;
        fill: white;
        opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)

def show_signup_page():
    """Display the professional sign up page"""
    apply_auth_styles()
    
    st.markdown("""
    <div class="auth-container">
        <div class="auth-card">
            <div class="auth-header">
                <div class="icon-wrapper">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
                    </svg>
                </div>
                <h1>Create Account</h1>
                <p>Join us to track and manage your stress levels</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Form container with white background
    st.markdown('<div class="auth-form">', unsafe_allow_html=True)
    
    with st.form("signup_form", clear_on_submit=False):
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            username = st.text_input(
                "👤 Username", 
                placeholder="Enter your username",
                help="Choose a unique username"
            )
            email = st.text_input(
                "📧 Email", 
                placeholder="Enter your email address",
                help="We'll never share your email"
            )
            password = st.text_input(
                "🔒 Password", 
                type="password", 
                placeholder="Create a strong password",
                help="Minimum 6 characters"
            )
            confirm_password = st.text_input(
                "🔒 Confirm Password", 
                type="password", 
                placeholder="Re-enter your password"
            )
            
            submitted = st.form_submit_button("🚀 Create Account", use_container_width=True)
        
        if submitted:
            if not username or not email or not password:
                st.error("⚠️ Please fill in all fields")
            elif len(username) < 3:
                st.error("⚠️ Username must be at least 3 characters long")
            elif "@" not in email or "." not in email:
                st.error("⚠️ Please enter a valid email address")
            elif password != confirm_password:
                st.error("⚠️ Passwords do not match")
            elif len(password) < 6:
                st.error("⚠️ Password must be at least 6 characters long")
            else:
                with st.spinner("Creating your account..."):
                    success, error_message = create_user(username, email, password)
                    if success:
                        st.success("✅ Account created successfully! Redirecting to sign in...")
                        st.session_state.show_signup = False
                        st.balloons()
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error(f"❌ {error_message}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Switch to sign in
    st.markdown("""
    <div class="auth-switch">
        <p>Already have an account?</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        if st.button("Sign In Instead", use_container_width=True, key="switch_to_signin"):
            st.session_state.show_signup = False
            st.rerun()

def show_signin_page():
    """Display the professional sign in page"""
    apply_auth_styles()
    
    st.markdown("""
    <div class="auth-container">
        <div class="auth-card">
            <div class="auth-header">
                <div class="icon-wrapper">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                    </svg>
                </div>
                <h1>Welcome Back</h1>
                <p>Sign in to access your stress detection dashboard</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Form container with white background
    st.markdown('<div class="auth-form">', unsafe_allow_html=True)
    
    with st.form("signin_form", clear_on_submit=False):
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            username = st.text_input(
                "👤 Username", 
                placeholder="Enter your username",
                help="Enter your registered username"
            )
            password = st.text_input(
                "🔒 Password", 
                type="password", 
                placeholder="Enter your password",
                help="Enter your account password"
            )
            
            submitted = st.form_submit_button("🔐 Sign In", use_container_width=True)
        
        if submitted:
            if not username or not password:
                st.error("⚠️ Please enter both username and password")
            else:
                with st.spinner("Signing you in..."):
                    user = authenticate_user(username, password)
                    if user:
                        st.success(f"✅ Welcome back, {user['username']}!")
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password. Please try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Switch to sign up
    st.markdown("""
    <div class="auth-switch">
        <p>Don't have an account yet?</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        if st.button("Create Account", use_container_width=True, key="switch_to_signup"):
            st.session_state.show_signup = True
            st.rerun()

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

