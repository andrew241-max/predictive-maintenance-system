"""
Страница анализа данных и обучения модели
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve, f1_score
)
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# Настройка стилей
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Константы
NUMERICAL_FEATURES = ['Air temperature [K]', 'Process temperature [K]',
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
FEATURE_NAMES = ['Type', 'Air temperature [K]', 'Process temperature [K]',
                 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']


@st.cache_data
def load_data():
    """
    Загрузка данных через ucimlrepo
    """
    try:
        from ucimlrepo import fetch_ucirepo

        with st.spinner('Загрузка данных из UCI Repository...'):
            # Загрузка датасета
            dataset = fetch_ucirepo(id=601)

            # Объединение признаков и целевых переменных
            data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

            return data
    except Exception as e:
        st.error(f"Ошибка загрузки данных через ucimlrepo: {e}")
        st.info("Пожалуйста, загрузите CSV файл вручную через интерфейс")
        return None


@st.cache_data
def preprocess_data(data):
    """
    Предобработка данных
    """
    df = data.copy()

    # Удаление ненужных столбцов
    columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Преобразование категориальной переменной Type в числовую
    if 'Type' in df.columns:
        le = LabelEncoder()
        df['Type'] = le.fit_transform(df['Type'])
        st.session_state['type_encoder'] = le

    # Проверка на пропущенные значения
    if df.isnull().sum().sum() > 0:
        st.warning("Обнаружены пропущенные значения. Выполняется заполнение медианой.")
        df = df.fillna(df.median())

    return df


def scale_features(df, fit=True):
    """
    Масштабирование числовых признаков
    """
    features = [f for f in NUMERICAL_FEATURES if f in df.columns]

    if fit:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        st.session_state['scaler'] = scaler
    else:
        scaler = st.session_state.get('scaler', StandardScaler())
        df[features] = scaler.transform(df[features])

    return df


def train_models(X_train, y_train):
    """
    Обучение моделей машинного обучения
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False,
                                 eval_metric='logloss'),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True, class_weight='balanced')
    }

    trained_models = {}
    for name, model in models.items():
        with st.spinner(f'Обучение {name}...'):
            model.fit(X_train, y_train)
            trained_models[name] = model

    return trained_models


def evaluate_model(model, X_test, y_test, model_name):
    """
    Оценка модели и возврат метрик
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        metrics['y_pred_proba'] = y_pred_proba
    else:
        metrics['roc_auc'] = None

    return metrics


def plot_confusion_matrix(cm, title="Матрица ошибок"):
    """
    Визуализация матрицы ошибок
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Нет отказа (0)', 'Отказ (1)'],
                yticklabels=['Нет отказа (0)', 'Отказ (1)'])
    ax.set_xlabel('Предсказанные значения')
    ax.set_ylabel('Фактические значения')
    ax.set_title(title)
    return fig


def plot_roc_curves(models_metrics, X_test, y_test):
    """
    Визуализация ROC-кривых для всех моделей
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, metrics in models_metrics.items():
        if metrics.get('y_pred_proba') is not None:
            fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
            auc = metrics['roc_auc']
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Случайное угадывание', linewidth=1)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC-кривые моделей классификации')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    return fig


def plot_feature_importance(model, feature_names):
    """
    Визуализация важности признаков для Random Forest и XGBoost
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(importances)), importances[indices])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_xlabel('Признаки')
        ax.set_ylabel('Важность')
        ax.set_title('Важность признаков')
        return fig
    return None


def analysis_and_model_page():
    """
    Основная страница приложения
    """
    st.title("🔧 Предиктивное обслуживание оборудования")
    st.markdown("""
    Данное приложение использует машинное обучение для прогнозирования отказов оборудования.
    Загрузите данные, обучите модель и получите предсказания.
    """)

    # Инициализация состояния сессии
    if 'models' not in st.session_state:
        st.session_state.models = None
    if 'models_metrics' not in st.session_state:
        st.session_state.models_metrics = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'trained' not in st.session_state:
        st.session_state.trained = False

    # Сайдбар для управления данными
    with st.sidebar:
        st.header("📁 Управление данными")

        # Выбор источника данных
        data_source = st.radio(
            "Источник данных:",
            ["Загрузить из UCI", "Загрузить CSV файл"]
        )

        data = None

        if data_source == "Загрузить из UCI":
            if st.button("📥 Загрузить данные из UCI", type="primary"):
                data = load_data()
                if data is not None:
                    st.session_state.raw_data = data
                    st.session_state.data_loaded = True
                    st.success("✅ Данные успешно загружены!")
        else:
            uploaded_file = st.file_uploader(
                "Выберите CSV файл с данными",
                type="csv",
                help="Файл должен содержать данные о состоянии оборудования"
            )
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.session_state.raw_data = data
                st.session_state.data_loaded = True
                st.success("✅ Файл успешно загружен!")

        # Отображение информации о данных
        if st.session_state.data_loaded and 'raw_data' in st.session_state:
            st.divider()
            st.header("📊 Информация о данных")
            st.metric("Количество записей", len(st.session_state.raw_data))
            st.metric("Количество признаков", len(st.session_state.raw_data.columns))

            if st.button("🔄 Предобработать данные", type="primary"):
                with st.spinner("Выполняется предобработка данных..."):
                    processed_data = preprocess_data(st.session_state.raw_data)
                    st.session_state.processed_data = processed_data
                    st.success("✅ Данные предобработаны!")

    # Основная область - вкладки
    if st.session_state.data_loaded and 'processed_data' in st.session_state:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Анализ данных",
            "🤖 Обучение моделей",
            "📊 Результаты",
            "🎯 Предсказания",
            "📋 Отчет"
        ])

        # Вкладка 1: Анализ данных
        with tab1:
            st.header("Анализ данных")

            # Предпросмотр данных
            st.subheader("Предпросмотр данных")
            st.dataframe(st.session_state.processed_data.head(10), use_container_width=True)

            # Статистика
            st.subheader("Статистическое описание")
            st.dataframe(st.session_state.processed_data.describe(), use_container_width=True)

            # Распределение целевой переменной
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Распределение отказов")
                target_col = 'Machine failure' if 'Machine failure' in st.session_state.processed_data.columns else 'Target'
                failure_counts = st.session_state.processed_data[target_col].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#2ecc71', '#e74c3c']
                ax.pie(failure_counts.values, labels=['Нет отказа', 'Отказ'],
                       autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title('Распределение целевой переменной')
                st.pyplot(fig)

                # Дисбаланс классов
                imbalance_ratio = failure_counts[0] / failure_counts[1] if failure_counts[1] > 0 else float('inf')
                st.info(f"Соотношение классов (0/1): {imbalance_ratio:.2f}:1")
                if imbalance_ratio > 10:
                    st.warning(
                        "⚠️ Обнаружен сильный дисбаланс классов. Рекомендуется использовать class_weight='balanced'")

            with col2:
                st.subheader("Корреляционная матрица")
                numeric_cols = st.session_state.processed_data.select_dtypes(include=[np.number]).columns
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(st.session_state.processed_data[numeric_cols].corr(),
                            annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                ax.set_title('Матрица корреляции признаков')
                st.pyplot(fig)

            # Распределение признаков
            st.subheader("Распределение числовых признаков")
            features_to_plot = [f for f in NUMERICAL_FEATURES if f in st.session_state.processed_data.columns]
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            for idx, feature in enumerate(features_to_plot):
                if idx < len(axes):
                    st.session_state.processed_data[feature].hist(bins=30, ax=axes[idx], color='#3498db',
                                                                  edgecolor='black')
                    axes[idx].set_title(feature)
                    axes[idx].set_xlabel('Значение')
                    axes[idx].set_ylabel('Частота')
            for idx in range(len(features_to_plot), len(axes)):
                axes[idx].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        # Вкладка 2: Обучение моделей
        with tab2:
            st.header("Обучение моделей машинного обучения")

            # Масштабирование
            scale_option = st.checkbox("Выполнить масштабирование данных", value=True)

            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Размер тестовой выборки", 0.1, 0.4, 0.2, 0.05)
            with col2:
                models_to_train = st.multiselect(
                    "Выберите модели для обучения",
                    ["Logistic Regression", "Random Forest", "XGBoost", "SVM"],
                    default=["Logistic Regression", "Random Forest"]
                )

            if st.button("🚀 Обучить выбранные модели", type="primary"):
                with st.spinner("Подготовка данных и обучение моделей..."):
                    # Подготовка данных
                    target_col = 'Machine failure' if 'Machine failure' in st.session_state.processed_data.columns else 'Target'
                    X = st.session_state.processed_data.drop(columns=[target_col])
                    y = st.session_state.processed_data[target_col]

                    # Разделение данных
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )

                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test

                    # Масштабирование
                    if scale_option:
                        X_train_scaled = scale_features(X_train.copy(), fit=True)
                        X_test_scaled = scale_features(X_test.copy(), fit=False)
                    else:
                        X_train_scaled = X_train
                        X_test_scaled = X_test

                    # Обучение выбранных моделей
                    all_models = {
                        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42,
                                                                  class_weight='balanced'),
                        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42,
                                                                class_weight='balanced'),
                        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42,
                                                 use_label_encoder=False, eval_metric='logloss'),
                        'SVM': SVC(kernel='rbf', random_state=42, probability=True, class_weight='balanced')
                    }

                    models_to_train_dict = {name: all_models[name] for name in models_to_train}

                    trained_models = {}
                    models_metrics = {}

                    for name, model in models_to_train_dict.items():
                        progress_text = f"Обучение {name}..."
                        progress_bar = st.progress(0, text=progress_text)
                        model.fit(X_train_scaled, y_train)
                        metrics = evaluate_model(model, X_test_scaled, y_test, name)
                        trained_models[name] = model
                        models_metrics[name] = metrics
                        progress_bar.progress(100, text=f"✅ {name} обучена!")
                        st.success(f"✅ {name} - Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

                    st.session_state.models = trained_models
                    st.session_state.models_metrics = models_metrics
                    st.session_state.trained = True
                    st.session_state.scaled = scale_option

                    st.balloons()
                    st.success("🎉 Все модели успешно обучены!")

        # Вкладка 3: Результаты
        with tab3:
            st.header("Оценка качества моделей")

            if st.session_state.trained and st.session_state.models_metrics:
                # Метрики
                st.subheader("Сравнение метрик")
                metrics_df = pd.DataFrame([
                    {
                        'Модель': name,
                        'Accuracy': metrics['accuracy'],
                        'F1-Score': metrics['f1'],
                        'ROC-AUC': metrics['roc_auc'] if metrics['roc_auc'] else 'N/A'
                    }
                    for name, metrics in st.session_state.models_metrics.items()
                ])
                st.dataframe(metrics_df.style.highlight_max(color='lightgreen'), use_container_width=True)

                # ROC-кривые
                st.subheader("ROC-кривые")
                fig_roc = plot_roc_curves(st.session_state.models_metrics, st.session_state.X_test,
                                          st.session_state.y_test)
                st.pyplot(fig_roc)

                # Матрицы ошибок для каждой модели
                st.subheader("Матрицы ошибок")
                cols = st.columns(len(st.session_state.models_metrics))
                for idx, (name, metrics) in enumerate(st.session_state.models_metrics.items()):
                    with cols[idx % len(cols)]:
                        st.markdown(f"**{name}**")
                        fig_cm = plot_confusion_matrix(metrics['confusion_matrix'], f"{name}")
                        st.pyplot(fig_cm)

                # Важность признаков (для Random Forest и XGBoost)
                st.subheader("Важность признаков")
                feature_names = st.session_state.X_train.columns.tolist()

                rf_name = next((name for name in st.session_state.models.keys() if 'Random Forest' in name), None)
                xgb_name = next((name for name in st.session_state.models.keys() if 'XGBoost' in name), None)

                col1, col2 = st.columns(2)
                if rf_name and st.session_state.models[rf_name]:
                    with col1:
                        fig_rf = plot_feature_importance(st.session_state.models[rf_name], feature_names)
                        if fig_rf:
                            st.pyplot(fig_rf)
                        else:
                            st.info("Модель Random Forest не поддерживает отображение важности признаков")

                if xgb_name and st.session_state.models[xgb_name]:
                    with col2:
                        fig_xgb = plot_feature_importance(st.session_state.models[xgb_name], feature_names)
                        if fig_xgb:
                            st.pyplot(fig_xgb)
                        else:
                            st.info("Модель XGBoost не поддерживает отображение важности признаков")

                # Детальный отчет по лучшей модели
                st.subheader("🏆 Лучшая модель")
                best_model_name = max(st.session_state.models_metrics.items(), key=lambda x: x[1]['accuracy'])[0]
                best_metrics = st.session_state.models_metrics[best_model_name]

                st.markdown(f"**Лучшая модель по Accuracy:** `{best_model_name}`")
                st.markdown(f"**Accuracy:** {best_metrics['accuracy']:.4f}")
                st.markdown(f"**F1-Score:** {best_metrics['f1']:.4f}")
                if best_metrics['roc_auc']:
                    st.markdown(f"**ROC-AUC:** {best_metrics['roc_auc']:.4f}")

                st.markdown("**Classification Report:**")
                st.code(classification_report(
                    st.session_state.y_test,
                    st.session_state.models[best_model_name].predict(
                        st.session_state.X_test if not st.session_state.get('scaled', False)
                        else scale_features(st.session_state.X_test.copy(), fit=False))
                ))
            else:
                st.info("👈 Пожалуйста, обучите модели на вкладке 'Обучение моделей'")

        # Вкладка 4: Предсказания
        with tab4:
            st.header("🎯 Предсказание отказов для новых данных")

            if st.session_state.trained and st.session_state.models:
                # Выбор модели
                model_names = list(st.session_state.models.keys())
                selected_model = st.selectbox("Выберите модель для предсказания", model_names)

                # Форма ввода данных
                with st.form("prediction_form"):
                    st.subheader("Введите параметры оборудования")

                    col1, col2 = st.columns(2)
                    with col1:
                        product_type = st.selectbox("Тип продукта", ["L (Low)", "M (Medium)", "H (High)"])
                        type_map = {"L (Low)": 0, "M (Medium)": 1, "H (High)": 2}
                        type_encoded = type_map[product_type]

                        air_temp = st.number_input(
                            "Температура воздуха [K]",
                            min_value=250.0, max_value=350.0, value=300.0, step=0.5
                        )

                        process_temp = st.number_input(
                            "Температура процесса [K]",
                            min_value=250.0, max_value=380.0, value=310.0, step=0.5
                        )

                        rotational_speed = st.number_input(
                            "Скорость вращения [rpm]",
                            min_value=100, max_value=3000, value=1500, step=10
                        )

                    with col2:
                        torque = st.number_input(
                            "Крутящий момент [Nm]",
                            min_value=0.0, max_value=150.0, value=40.0, step=1.0
                        )

                        tool_wear = st.number_input(
                            "Износ инструмента [min]",
                            min_value=0, max_value=300, value=100, step=5
                        )

                    submit = st.form_submit_button("🔮 Выполнить предсказание", type="primary")

                    if submit:
                        # Создание DataFrame с входными данными
                        input_data = pd.DataFrame([[
                            type_encoded, air_temp, process_temp, rotational_speed, torque, tool_wear
                        ]], columns=FEATURE_NAMES)

                        # Масштабирование если нужно
                        if st.session_state.get('scaled', False):
                            if 'scaler' in st.session_state:
                                numerical_cols = [f for f in NUMERICAL_FEATURES if f in input_data.columns]
                                input_data[numerical_cols] = st.session_state.scaler.transform(
                                    input_data[numerical_cols])

                        # Предсказание
                        model = st.session_state.models[selected_model]
                        prediction = model.predict(input_data)[0]

                        if hasattr(model, "predict_proba"):
                            probability = model.predict_proba(input_data)[0]
                            prob_failure = probability[1]
                        else:
                            prob_failure = None

                        # Отображение результата
                        st.divider()
                        st.subheader("📊 Результат предсказания")

                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if prediction == 1:
                                st.error("⚠️ ПРЕДУПРЕЖДЕНИЕ: Ожидается отказ оборудования!")
                                st.markdown("""
                                ### Рекомендации:
                                - 🔧 Проведите внеплановое техническое обслуживание
                                - 📊 Проверьте все критические параметры
                                - 👨‍🔧 Вызовите сервисную бригаду
                                """)
                            else:
                                st.success("✅ Оборудование в нормальном состоянии")
                                st.markdown("""
                                ### Рекомендации:
                                - 📋 Продолжайте плановое обслуживание
                                - 📈 Регулярно мониторьте параметры
                                """)

                            if prob_failure is not None:
                                st.metric("Вероятность отказа", f"{prob_failure:.2%}")

                                # Индикатор вероятности
                                if prob_failure < 0.3:
                                    st.progress(prob_failure, text="Низкий риск")
                                elif prob_failure < 0.7:
                                    st.progress(prob_failure, text="Средний риск")
                                else:
                                    st.progress(prob_failure, text="Высокий риск")
            else:
                st.info("👈 Пожалуйста, сначала обучите модели на вкладке 'Обучение моделей'")

        # Вкладка 5: Отчет
        with tab5:
            st.header("📋 Отчет о работе приложения")

            if st.session_state.trained:
                # Общая информация
                st.subheader("Информация о проекте")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Размер обучающей выборки",
                              len(st.session_state.X_train) if st.session_state.X_train is not None else "N/A")
                with col2:
                    st.metric("Размер тестовой выборки",
                              len(st.session_state.X_test) if st.session_state.X_test is not None else "N/A")
                with col3:
                    st.metric("Количество обученных моделей",
                              len(st.session_state.models) if st.session_state.models else 0)

                # Краткое резюме
                st.subheader("Резюме по моделям")

                best_model = max(st.session_state.models_metrics.items(), key=lambda x: x[1]['accuracy'])

                st.markdown(f"""
                ### 🏆 Лучшая модель: **{best_model[0]}**
                - **Точность (Accuracy):** {best_model[1]['accuracy']:.2%}
                - **F1-мера:** {best_model[1]['f1']:.4f}
                - **ROC-AUC:** {best_model[1]['roc_auc']:.4f}
                """)

                # Ключевые выводы
                st.subheader("Ключевые выводы")

                st.markdown("""
                #### О данных:
                - Обнаружен дисбаланс классов, что характерно для задач предиктивного обслуживания
                - Ключевые признаки: износ инструмента, крутящий момент и скорость вращения наиболее сильно коррелируют с отказами

                #### О моделях:
                - Random Forest и XGBoost показывают лучшие результаты благодаря способности работать с нелинейными зависимостями
                - ROC-AUC > 0.9 указывает на хорошую разделяющую способность моделей

                #### Рекомендации:
                - Использовать ансамблевые методы (Random Forest / XGBoost) для получения наилучших результатов
                - Регулярно переобучать модель на новых данных
                - Мониторить качество предсказаний в production
                """)
            else:
                st.info("👈 Пожалуйста, обучите модели для просмотра отчета")

    else:
        # Отображение инструкции
        st.info("👈 Пожалуйста, загрузите данные через боковую панель")
        st.markdown("""
        ### 📋 Инструкция по использованию:
        1. **Загрузите данные** через боковую панель (из UCI или CSV файл)
        2. **Предобработайте данные** кнопкой "Предобработать данные"
        3. **Исследуйте данные** на вкладке "Анализ данных"
        4. **Обучите модели** на вкладке "Обучение моделей"
        5. **Оцените результаты** на вкладке "Результаты"
        6. **Сделайте предсказания** для новых данных
        """)