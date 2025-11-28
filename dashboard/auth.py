import streamlit as st
from database import create_user, authenticate_user, init_database

def show_signup_page():
    """Display the sign up page"""
    st.title("Sign Up")
    st.markdown("Create a new account to track your stress levels")
    
    with st.form("signup_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
        
        submitted = st.form_submit_button("Sign Up", use_container_width=True)
        
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
                    st.rerun()
                else:
                    st.error(f"Error: {error_message}")
    
    st.markdown("---")
    st.markdown("Already have an account?")
    if st.button("Sign In", use_container_width=True):
        st.session_state.show_signup = False
        st.rerun()

def show_signin_page():
    """Display the sign in page"""
    st.title("Sign In")
    st.markdown("Sign in to access your stress detection dashboard")
    
    with st.form("signin_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        submitted = st.form_submit_button("Sign In", use_container_width=True)
        
        if submitted:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                user = authenticate_user(username, password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.success(f"Welcome back, {user['username']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    st.markdown("---")
    st.markdown("Don't have an account?")
    if st.button("Sign Up", use_container_width=True):
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

