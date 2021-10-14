import streamlit as st

from demos.statistics import anscombe_demo

st.set_page_config(layout="wide", page_title='Anscombe Demo')
anscombe_demo()
