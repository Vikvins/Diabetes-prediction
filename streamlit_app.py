import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os 

st.set_page_config(
    page_title="Предсказание диабета", 
    layout="centered",
    initial_sidebar_state="auto"
)



MODEL_FILENAME = 'diabetes_model.pkl'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        st.error(f"❌ Ошибка: Файл модели '{MODEL_FILENAME}' не найден.")
        st.info("Пожалуйста, убедитесь, что вы сохранили модель в Jupiter Notebook командой joblib.dump(lrm, 'diabetes_model.pkl') и что файл находится в той же папке.")
        return None
    
    try:
        model = joblib.load(MODEL_FILENAME)
        return model
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке модели: {e}")
        return None

lrn = load_model()


st.title("🩺 Приложение для предсказания диабета")

st.write("Введите следующие параметры для оценки риска развития диабета:")

if lrn is not None:
    
    FEATURE_NAMES = [
        'Pregnancies', 
        'Glucose', 
        'BloodPressure', 
        'SkinThickness', 
        'Insulin', 
        'BMI', 
        'DiabetesPedigreeFunction', 
        'Age'
    ]

    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input(
            '1. Количество беременностей', 
            min_value=0, max_value=20, value=1, step=1,
            help="Количество беременностей"
        )
        glucose = st.number_input(
            '2. Уровень глюкозы (мг/дл)', 
            min_value=0.0, max_value=200.0, value=120.0, step=0.1,
            help="Концентрация глюкозы в плазме"
        )
        blood_pressure = st.number_input(
            '3. Кровяное давление (мм рт. ст.)', 
            min_value=0.0, max_value=150.0, value=70.0, step=0.1,
            help="Диастолическое артериальное давление"
        )
        skin_thickness = st.number_input(
            '4. Толщина кожной складки (мм)', 
            min_value=0.0, max_value=100.0, value=25.0, step=0.1,
            help="Толщина кожной складки на трицепсе"
        )
        
    with col2:
        insulin = st.number_input(
            '5. Уровень инсулина (мкМЕ/мл)', 
            min_value=0.0, max_value=900.0, value=0.0, step=0.1,
            help="2-часовой сывороточный инсулин"
        )
        bmi = st.number_input(
            '6. Индекс массы тела (кг/м²)', 
            min_value=0.0, max_value=70.0, value=25.0, step=0.1,
            help="Индекс массы тела"
        )
        dpf = st.number_input(
            '7. Функция родословной диабета', 
            min_value=0.0, max_value=2.5, value=0.4, step=0.001, format="%.3f",
            help="Оценка генетического риска"
        )
        age = st.number_input(
            '8. Возраст', 
            min_value=0, max_value=120, value=30, step=1,
            help="Возраст человека в годах"
        )
        
    st.markdown(" ")
    
    predict_button = st.button("🔍 Получить предсказание", type="primary", use_container_width=True)

    if predict_button:
        input_data = [
            pregnancies, glucose, blood_pressure, skin_thickness, 
            insulin, bmi, dpf, age
        ]
        
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        prediction = lrn.predict(input_df)[0] 
        
        st.header("Результат анализа:")
        
        if prediction == 1:
            st.error("🔴 **ПОЛОЖИТЕЛЬНЫЙ РЕЗУЛЬТАТ**")
            st.markdown("Модель предсказывает **высокий риск** диабета. Рекомендуется консультация врача.")
        else:
            st.success("🟢 **ОТРИЦАТЕЛЬНЫЙ РЕЗУЛЬТАТ**")
            st.markdown("Модель предсказывает **низкий риск** диабета. Продолжайте следить за здоровьем.")
            
        probability = lrn.predict_proba(input_df)[0]
        st.caption(f"Вероятность диабета (класс 1): **{probability[1]*100:.2f}%**")
        
    st.markdown("---")
    st.warning("⚠️ **ВНИМАНИЕ:** Этот сайт является демонстрацией модели машинного обучения и не является медицинским инструментом. Полученные результаты не заменяют консультацию специалиста и профессиональную диагностику.")


st.sidebar.title("Инструкции")
st.sidebar.info("1. Сохраните этот код как `streamlit_app.py`.\n2. Убедитесь, что `diabetes_model.pkl` находится рядом.\n3. Запустите в терминале: `streamlit run streamlit_app.py`")
