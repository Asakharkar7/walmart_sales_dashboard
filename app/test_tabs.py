import streamlit as st

st.title("Tab Test")
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

with tab1:
    st.write("This is tab 1")

with tab2:
    st.write("This is tab 2")
