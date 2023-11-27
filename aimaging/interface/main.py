import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import about, home, result

st.set_page_config(
    page_title="Organ Disease Detector 🔍"
    )

class MultiApp:
    def __init__(self):
        self.apps = []
    def add_app(self, title, function):
        self.apps.append({
            "title": title,
            "function": function
        })

    def run():

        st.markdown(
            """
            <style>
                body {
                    background-color: black;
                }
            </style>
            """,
            unsafe_allow_html=True)

        with st.sidebar:
            app = option_menu(
                menu_title="Disease detector ",
                options=["Home", "Result", "About"],
                icons=["house", "list-task", "info-circle-fill"],
                menu_icon="clipboard-pulse",
                default_index=0,
                styles={"container": {"padding": "5!important", "background-color": 'black'},
                        "icon": {"color": "white", "font-size": "23px"},
                        "menu-title": {"color": "white", "font-size": "25px", "margin": "0px", "background-color": "black",
                                       "font-weight": "bold"},
                        "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px",
                                     "--hover-color": "#996500"},
                        "nav-link-selected": {"background-color": "#910a0a"}, }
            )

        if app == "Home":
            home.app()
            # Check if the scanning process has been initiated
            if st.session_state.get('scan_button_clicked', False):
                # Automatically switch to the Result page
                result.app()

        if app == "Result":
            result.app()
        if app == "About":
            about.app()


    run()
