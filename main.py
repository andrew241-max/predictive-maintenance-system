"""
Основной файл Streamlit-приложения для прогнозирования отказов оборудования
"""
import streamlit as st

# Настройка страницы
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Импорт страниц
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

# Настройка навигации
pages = {
    "Анализ и модель": st.Page(analysis_and_model_page, title="📊 Анализ и модель"),
    "Презентация": st.Page(presentation_page, title="📽️ Презентация проекта"),
}

# Отображение навигации
current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()