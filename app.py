# -*- coding: utf-8 -*-

import streamlit as st
from streamlit_option_menu import option_menu
import utils

def main():
    """
        Main function to run the Streamlit app.
    """

    st.set_page_config(page_title="Forest Fire ")
    # Streamlit 앱 실행
    with st.sidebar:
        selected = option_menu("Main Menu", ["INTRO", "DATA", "EDA", "STAT", "ML", "DL"],
                               icons=["house", "card-checklist", "bar-chart", "clipboard-data", "gear", "gear"],
                               menu_icon="cast",
                               default_index=0,
                               orientation="vertical")

if __name__ == "__main__":
    main()