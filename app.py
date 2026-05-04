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

# Создание боковой панели для навигации
st.sidebar.title("⚙️ Навигация")
st.sidebar.markdown("---")

# Выбор страницы через radio button
page = st.sidebar.radio(
    "Выберите раздел:",
    ["📊 Анализ и модель", "📽️ Презентация проекта"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **О проекте:**
    Система предиктивного обслуживания
    оборудования на основе ML

    **Технологии:**
    - Streamlit
    - Scikit-learn
    - Pandas
    - XGBoost
    """
)

# Отображение выбранной страницы
if page == "📊 Анализ и модель":
    analysis_and_model_page()
else:
    presentation_page()