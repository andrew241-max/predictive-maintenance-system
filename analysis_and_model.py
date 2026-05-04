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
    roc_curve, roc_auc_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

# Попытка импорта XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Настройка стилей
plt.style.use('default')
sns.set_palette("husl")

# Константы
NUMERICAL_FEATURES = ['Air temperature', 'Process temperature',
                       'Rotational speed', 'Torque', 'Tool wear']
FEATURE_NAMES = ['Type', 'Air temperature', 'Process temperature',
                 'Rotational speed', 'Torque', 'Tool wear']

def rename_columns(df):
    """
    Переименовывает колонки для единообразия
    """
    rename_map = {
        'Air temperature [K]': 'Air temperature',
        'Process temperature [K]': 'Process temperature',
        'Rotational speed [rpm]': 'Rotational speed',
        'Torque [Nm]': 'Torque',
        'Tool wear [min]': 'Tool wear'
    }

    for old_name, new_name in rename_map.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})

    return df

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

def preprocess_data(data):
    """
    Предобработка данных
    """
    df = data.copy()

    # Переименовываем колонки для единообразия
    df = rename_columns(df)

    # Удаление ненужных столбцов
    columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)

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
    # Проверяем, какие числовые признаки есть в данных
    features = [f for f in NUMERICAL_FEATURES if f in df.columns]

    # Если нет признаков для масштабирования, возвращаем исходный DataFrame
    if len(features) == 0:
        return df

    try:
        if fit:
            scaler = StandardScaler()
            # Масштабируем признаки
            scaled_data = scaler.fit_transform(df[features])
            df[features] = scaled_data
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = features
        else:
            if st.session_state.get('scaler') is not None:
                scaler = st.session_state.scaler
                scaled_data = scaler.transform(df[features])
                df[features] = scaled_data
    except Exception as e:
        st.error(f"Ошибка при масштабировании данных: {e}")

    return df

def train_models(X_train, y_train, model_names):
    """
    Обучение выбранных моделей машинного обучения
    """
    all_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True, class_weight='balanced')
    }

    if XGBOOST_AVAILABLE:
        all_models['XGBoost'] = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42,
                                              use_label_encoder=False, eval_metric='logloss')

    trained_models = {}
    progress_bar = st.progress(0)
    for idx, name in enumerate(model_names):
        if name in all_models:
            st.write(f"🔄 Обучение {name}...")
            model = all_models[name]
            model.fit(X_train, y_train)
            trained_models[name] = model
            progress_bar.progress((idx + 1) / len(model_names))

    progress_bar.empty()
    return trained_models

def evaluate_model(model, X_test, y_test):
    """
    Оценка модели и возврат метрик
    """
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred
    }

    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        metrics['y_pred_proba'] = y_pred_proba
    else:
        metrics['roc_auc'] = None
        metrics['y_pred_proba'] = None

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

def plot_roc_curves(models_metrics):
    """
    Визуализация ROC-кривых для всех моделей
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, metrics in models_metrics.items():
        if metrics.get('y_pred_proba') is not None and st.session_state.get('y_test') is not None:
            fpr, tpr, _ = roc_curve(st.session_state.y_test, metrics['y_pred_proba'])
            auc = metrics['roc_auc']
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Случайное угадывание', linewidth=1)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC-кривые моделей классификации')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    return fig

def plot_feature_importance(model, feature_names, model_name):
    """
    Визуализация важности признаков
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(importances)), importances[indices])
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Важность')
        ax.set_title(f'Важность признаков - {model_name}')
        ax.invert_yaxis()
        return fig
    return None

def analysis_and_model_page():
    """
    Основная страница приложения
    """
    st.title("🔧 Предиктивное обслуживание оборудования")
    st.markdown("---")

    # Инициализация session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    # Сайдбар для управления данными
    with st.sidebar:
        st.header("📁 Управление данными")

        # Выбор источника данных
        data_source = st.radio(
            "Источник данных:",
            ["Загрузить из UCI", "Загрузить CSV файл"]
        )

        if data_source == "Загрузить из UCI":
            if st.button("📥 Загрузить данные из UCI"):
                data = load_data()
                if data is not None:
                    st.session_state.raw_data = data
                    st.session_state.data_loaded = True
                    st.session_state.processed_data = None
                    st.session_state.trained = False
                    st.success("✅ Данные успешно загружены!")
        else:
            uploaded_file = st.file_uploader(
                "Выберите CSV файл с данными",
                type="csv"
            )
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.session_state.raw_data = data
                st.session_state.data_loaded = True
                st.session_state.processed_data = None
                st.session_state.trained = False
                st.success("✅ Файл успешно загружен!")

        # Отображение информации о данных
        if st.session_state.data_loaded and st.session_state.raw_data is not None:
            st.markdown("---")
            st.header("📊 Информация о данных")
            st.write(f"**Количество записей:** {len(st.session_state.raw_data)}")
            st.write(f"**Количество признаков:** {len(st.session_state.raw_data.columns)}")

            if st.button("🔄 Предобработать данные"):
                with st.spinner("Выполняется предобработка данных..."):
                    processed_data = preprocess_data(st.session_state.raw_data)
                    st.session_state.processed_data = processed_data
                    st.session_state.trained = False
                    st.success("✅ Данные предобработаны!")

    # Основная область - вкладки
    if st.session_state.data_loaded and st.session_state.processed_data is not None:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Анализ данных",
            "🤖 Обучение моделей",
            "📊 Результаты",
            "🎯 Предсказания"
        ])

        # Вкладка 1: Анализ данных
        with tab1:
            st.header("Анализ данных")

            # Предпросмотр данных
            st.subheader("Предпросмотр данных")
            st.dataframe(st.session_state.processed_data.head(10))

            # Статистика
            st.subheader("Статистическое описание")
            st.dataframe(st.session_state.processed_data.describe())

            # Распределение целевой переменной
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Распределение отказов")
                target_col = 'Machine failure' if 'Machine failure' in st.session_state.processed_data.columns else 'Target'
                if target_col in st.session_state.processed_data.columns:
                    failure_counts = st.session_state.processed_data[target_col].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['#2ecc71', '#e74c3c']
                    ax.pie(failure_counts.values, labels=['Нет отказа', 'Отказ'],
                           autopct='%1.1f%%', colors=colors, startangle=90)
                    ax.set_title('Распределение целевой переменной')
                    st.pyplot(fig)

                    # Дисбаланс классов
                    if len(failure_counts) > 1:
                        imbalance_ratio = failure_counts[0] / failure_counts[1]
                        st.info(f"Соотношение классов (0/1): {imbalance_ratio:.2f}:1")
                        if imbalance_ratio > 10:
                            st.warning("⚠️ Обнаружен сильный дисбаланс классов")

            with col2:
                st.subheader("Корреляционная матрица")
                numeric_cols = st.session_state.processed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(st.session_state.processed_data[numeric_cols].corr(),
                                annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                    ax.set_title('Матрица корреляции признаков')
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
                available_models = ["Logistic Regression", "Random Forest", "SVM"]
                if XGBOOST_AVAILABLE:
                    available_models.append("XGBoost")

                models_to_train = st.multiselect(
                    "Выберите модели для обучения",
                    available_models,
                    default=["Random Forest"]
                )

            if st.button("🚀 Обучить выбранные модели"):
                if len(models_to_train) == 0:
                    st.warning("Пожалуйста, выберите хотя бы одну модель для обучения")
                else:
                    with st.spinner("Подготовка данных и обучение моделей..."):
                        # Подготовка данных
                        target_col = 'Machine failure' if 'Machine failure' in st.session_state.processed_data.columns else 'Target'
                        X = st.session_state.processed_data.drop(columns=[target_col])
                        y = st.session_state.processed_data[target_col]

                        # Разделение данных
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y
                        )

                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test

                        # Масштабирование
                        if scale_option:
                            X_train_scaled = scale_features(X_train.copy(), fit=True)
                            X_test_scaled = scale_features(X_test.copy(), fit=False)
                        else:
                            X_train_scaled = X_train
                            X_test_scaled = X_test

                        # Обучение выбранных моделей
                        trained_models = train_models(X_train_scaled, y_train, models_to_train)

                        # Оценка моделей
                        models_metrics = {}
                        for name, model in trained_models.items():
                            metrics = evaluate_model(model, X_test_scaled, y_test)
                            models_metrics[name] = metrics
                            st.success(f"✅ {name} - Accuracy: {metrics['accuracy']:.4f}")

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
                metrics_data = []
                for name, metrics in st.session_state.models_metrics.items():
                    row = {
                        'Модель': name,
                        'Accuracy': f"{metrics['accuracy']:.4f}",
                        'F1-Score': f"{metrics['f1']:.4f}",
                    }
                    if metrics['roc_auc']:
                        row['ROC-AUC'] = f"{metrics['roc_auc']:.4f}"
                    else:
                        row['ROC-AUC'] = "N/A"
                    metrics_data.append(row)

                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df)

                # ROC-кривые
                if any(metrics.get('roc_auc') is not None for metrics in st.session_state.models_metrics.values()):
                    st.subheader("ROC-кривые")
                    fig_roc = plot_roc_curves(st.session_state.models_metrics)
                    st.pyplot(fig_roc)

                # Матрицы ошибок
                st.subheader("Матрицы ошибок")
                model_names = list(st.session_state.models_metrics.keys())
                cols = st.columns(min(len(model_names), 2))

                for idx, name in enumerate(model_names):
                    with cols[idx % len(cols)]:
                        st.markdown(f"**{name}**")
                        fig_cm = plot_confusion_matrix(
                            st.session_state.models_metrics[name]['confusion_matrix'],
                            f"{name}"
                        )
                        st.pyplot(fig_cm)

                # Важность признаков
                if st.session_state.X_test is not None:
                    st.subheader("Важность признаков")
                    feature_names = st.session_state.X_test.columns.tolist()

                    col1, col2 = st.columns(2)

                    # Random Forest
                    if 'Random Forest' in st.session_state.models:
                        with col1:
                            fig_rf = plot_feature_importance(
                                st.session_state.models['Random Forest'],
                                feature_names,
                                "Random Forest"
                            )
                            if fig_rf:
                                st.pyplot(fig_rf)

                    # XGBoost
                    if XGBOOST_AVAILABLE and 'XGBoost' in st.session_state.models:
                        with col2:
                            fig_xgb = plot_feature_importance(
                                st.session_state.models['XGBoost'],
                                feature_names,
                                "XGBoost"
                            )
                            if fig_xgb:
                                st.pyplot(fig_xgb)

                # Детальный отчет по лучшей модели
                st.subheader("🏆 Лучшая модель")
                best_model_name = max(st.session_state.models_metrics.items(), key=lambda x: x[1]['accuracy'])[0]
                best_metrics = st.session_state.models_metrics[best_model_name]

                st.markdown(f"**Лучшая модель по Accuracy:** `{best_model_name}`")
                st.markdown(f"**Accuracy:** {best_metrics['accuracy']:.4f}")
                st.markdown(f"**F1-Score:** {best_metrics['f1']:.4f}")
                if best_metrics['roc_auc']:
                    st.markdown(f"**ROC-AUC:** {best_metrics['roc_auc']:.4f}")

                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(st.session_state.y_test, best_metrics['y_pred'])
                st.text(report)
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
                st.subheader("Введите параметры оборудования:")

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

                with col2:
                    rotational_speed = st.number_input(
                        "Скорость вращения [rpm]",
                        min_value=100, max_value=3000, value=1500, step=10
                    )

                    torque = st.number_input(
                        "Крутящий момент [Nm]",
                        min_value=0.0, max_value=150.0, value=40.0, step=1.0
                    )

                    tool_wear = st.number_input(
                        "Износ инструмента [min]",
                        min_value=0, max_value=300, value=100, step=5
                    )

                if st.button("🔮 Выполнить предсказание"):
                    # Создание DataFrame с входными данными
                    input_data = pd.DataFrame([[
                        type_encoded, air_temp, process_temp, rotational_speed, torque, tool_wear
                    ]], columns=['Type', 'Air temperature', 'Process temperature',
                               'Rotational speed', 'Torque', 'Tool wear'])

                    # Масштабирование если нужно
                    if st.session_state.get('scaled', False) and st.session_state.get('scaler') is not None:
                        numerical_cols = ['Air temperature', 'Process temperature',
                                         'Rotational speed', 'Torque', 'Tool wear']
                        numerical_cols = [col for col in numerical_cols if col in input_data.columns]
                        if numerical_cols:
                            input_data[numerical_cols] = st.session_state.scaler.transform(input_data[numerical_cols])

                    # Предсказание
                    model = st.session_state.models[selected_model]
                    prediction = model.predict(input_data)[0]

                    if hasattr(model, "predict_proba"):
                        probability = model.predict_proba(input_data)[0]
                        prob_failure = probability[1]
                    else:
                        prob_failure = None

                    # Отображение результата
                    st.markdown("---")
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
                            st.write(f"**Вероятность отказа:** {prob_failure:.2%}")

                            # Отображаем прогресс-бар
                            st.progress(prob_failure)

                            # Отображаем текстовый индикатор риска отдельно
                            if prob_failure < 0.3:
                                st.success("🟢 Низкий риск отказа")
                            elif prob_failure < 0.7:
                                st.warning("🟡 Средний риск отказа")
                            else:
                                st.error("🔴 Высокий риск отказа")
            else:
                st.info("👈 Пожалуйста, сначала обучите модели на вкладке 'Обучение моделей'")

    elif st.session_state.data_loaded and st.session_state.processed_data is None:
        st.info("👈 Пожалуйста, нажмите кнопку 'Предобработать данные' в боковой панели")

    else:
        # Отображение инструкции
        st.info("👈 Пожалуйста, загрузите данные через боковую панель")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### 📋 Инструкция по использованию:
            
            1. **Загрузите данные** через боковую панель:
               - Из UCI Repository (нажмите кнопку)
               - Или загрузите CSV файл
               
            2. **Предобработайте данные** кнопкой в боковой панели
            
            3. **Исследуйте данные** на вкладке "Анализ данных"
            
            4. **Обучите модели** на вкладке "Обучение моделей"
            
            5. **Оцените результаты** на вкладке "Результаты"
            
            6. **Сделайте предсказания** для новых данных
            """)