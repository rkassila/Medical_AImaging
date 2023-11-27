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

        with st.sidebar:
            app = option_menu(
                menu_title = "Disease detector ",
                options = ["Home", "Result", "About"],
                icons = ["house", "list-task", "info-circle-fill"],
                menu_icon= "clipboard-pulse",
                default_index= 1,
                styles = {"container": {"padding": "5!important","background-color":'black'},
                          "icon": {"color": "white", "font-size": "23px"},
                          "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
                          "nav-link-selected": {"background-color": "#02ab21"},}
                )

        if app == "Home":
            home.app()
        if app == "Result":
            result.app()
        if app == "About":
            about.app()


    run()
