import streamlit as st

from utils import get_current_time_korea_formatted

st.set_page_config(
    page_title="Find Retailers",
    page_icon="🕵️‍♂️",
)

st.markdown("""
# Find Retailers
""")

if 'access_granted' not in st.session_state:
    st.session_state['access_granted'] = False

activation_key = st.text_input("페이지 접근을 위한 키값을 입력해 주세요:")
correct_key = st.secrets["ACTIVATION_KEY"]

if st.button("확인"):
    if (activation_key == correct_key):
        st.session_state['access_granted'] = True
        st.success("FindUs 페이지로 이동하셔서 인재를 검색해 보세요!")
        st.session_state['thread_id'] = get_current_time_korea_formatted()
    else:
        st.session_state['access_granted'] = False
        if activation_key != correct_key:
            st.error("올바른 키값을 입력하세요.")
