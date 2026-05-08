from __future__ import annotations

from urllib.request import urlopen

import streamlit as st
import streamlit.components.v1 as components


FRONTEND_URL = "http://127.0.0.1:5174"


def frontend_is_running() -> bool:
    try:
        with urlopen(FRONTEND_URL, timeout=1.5) as response:
            return response.status == 200
    except Exception:
        return False


st.set_page_config(
    page_title="多模态 RAG 证据检测页面",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container { padding: 0; max-width: 100%; }
      header[data-testid="stHeader"] { display: none; }
      div[data-testid="stToolbar"] { display: none; }
      iframe { display: block; }
    </style>
    """,
    unsafe_allow_html=True,
)

if frontend_is_running():
    components.iframe(FRONTEND_URL, height=950, scrolling=True)
else:
    st.error("新的 React 页面服务还没有启动。")
    st.code(
        "python scripts/13_export_frontend_data.py\n"
        "cd web && npm install && npm run build\n"
        "cd web/dist && python -m http.server 5174 --bind 127.0.0.1",
        language="bash",
    )
    st.link_button("打开 React 页面", FRONTEND_URL)
