import streamlit as st

st.set_page_config(
    page_title="Medicare Part D",
    layout="wide",
    initial_sidebar_state="collapsed",
)

dashboard = st.Page(
    "Med_D_dashboard.py",
    title="Part D Dashboard",
    icon=":material/insights:",
    default=True,
)
provider_search = st.Page(
    "pages/1_Provider_Search.py",
    title="Provider Search",
    icon=":material/search:",
)
part_b_drugs = st.Page(
    "pages/2_Part_B_Drugs.py",
    title="Med B Drugs Dashboard",
    icon=":material/medication:",
)

nav = st.navigation([dashboard, part_b_drugs, provider_search], position="top")
nav.run()
