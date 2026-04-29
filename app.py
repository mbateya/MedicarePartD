import streamlit as st

st.set_page_config(
    page_title="Medicare Part D",
    layout="wide",
    initial_sidebar_state="collapsed",
)

dashboard = st.Page(
    "Med_D_dashboard.py",
    title="Dashboard",
    icon=":material/insights:",
    default=True,
)
provider_search = st.Page(
    "pages/1_Provider_Search.py",
    title="Provider Search",
    icon=":material/search:",
)

nav = st.navigation([dashboard, provider_search], position="top")
nav.run()
