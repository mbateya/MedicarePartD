import streamlit as st

from dashboard_design import install_design_system

st.set_page_config(
    page_title="Medicare Part D",
    layout="wide",
    initial_sidebar_state="collapsed",
)

install_design_system()

dashboard = st.Page(
    "Med_D_dashboard.py",
    title="Med D Drugs Dashboard",
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
part_b_drugs_state = st.Page(
    "pages/3_Med_B_Drugs_State.py",
    title="Med B Drugs by State & Provider Specialty",
    icon=":material/map:",
)

nav = st.navigation(
    [dashboard, part_b_drugs, part_b_drugs_state, provider_search],
    position="top",
)
nav.run()
