import streamlit as st
import lib as lb

@st.cache_data(show_spinner=False)
def load_anfavea(path): 
    return lb.base_anfavea(path)

@st.cache_data(show_spinner=False)
def load_emplacamento(pattern): 
    return lb.carregar_emplacamento(pattern)