
# app.py
import numpy as np
import streamlit as st
import joblib
import tensorflow as tf
from collections import Counter

# Загрузка предобученных моделей и артефактов
model_shielding = tf.keras.models.load_model('model_ee.keras')
le_shielding = joblib.load('le_ee.pkl')
avg_eff_shielding = joblib.load('avg_eff_ee.pkl')

model_absorption = tf.keras.models.load_model('model_kp.keras')
le_absorption = joblib.load('le_kp.pkl')
avg_eff_absorption = joblib.load('avg_eff_kp.pkl')

model_reflection = tf.keras.models.load_model('model_ko.keras')
le_reflection = joblib.load('le_ko.pkl')
avg_eff_reflection = joblib.load('avg_eff_ko.pkl')

# Универсальная функция предсказания наиболее частого материала

def predict_most_frequent_material(model, le, freq_start, freq_end):
    freqs = np.arange(freq_start, freq_end + 1, 1)
    predictions = []
    for f in freqs:
        input_data = np.array([[f, min_val]])  # два признака
        pred = model.predict(input_data, verbose=0)
        idx = np.argmax(pred)
        predictions.append(idx)
    most_common_idx = Counter(predictions).most_common(1)[0][0]
    material = le.inverse_transform([most_common_idx])[0]
    confidence = predictions.count(most_common_idx) / len(predictions)
    return material, confidence

# Функции для формирования композитов

def create_shielding_composite(freq_start, freq_end, min_value):
    material, conf = predict_most_frequent_material(model_shielding, le_shielding, freq_start, freq_end)
    avg = avg_eff_shielding.get(material, 0)
    if avg < min_value:
        desc = f"Предсказанный материал — {material} (средняя {avg:.3f}) не удовлетворяет min={min_value}."
    else:
        desc = f"Предсказанный материал — {material} (средняя {avg:.3f}, уверенность {conf:.3f})."
    return {
        "1. Защитный внешний слой": "Полиуретановое покрытие",
        "2. Экранирующий металлический слой": desc,
        "3. Диэлектрический слой с армирующей сеткой": "Полиимид с армирующей сеткой",
        "4. Поглощающий внутренний слой (опционально)": "Графен",
        "5. Внутренний защитный слой": "Полиуретановое покрытие"
    }


def create_absorption_composite(freq_start, freq_end, min_value):
    material, conf = predict_most_frequent_material(model_absorption, le_absorption, freq_start, freq_end)
    avg = avg_eff_absorption.get(material, 0)
    if avg < min_value:
        desc = f"Предсказанный материал — {material} (среднее {avg:.3f}) не удовлетворяет min={min_value}."
    else:
        desc = f"Предсказанный материал — {material} (среднее {avg:.3f}, уверенность {conf:.3f})."
    return {
        "1. Внешний демпфирующий слой": "Полиуретан",
        "2. Магнитный слой на основе феррита": "Никель-цинковый феррит",
        "3. Поглощающий композитный слой": desc,
        "4. Полимерная прослойка (опционально для гибкости)": "Графен",
        "5. Внутренний структурный слой": "Стекловолокно"
    }


def create_reflection_composite(freq_start, freq_end, min_value):
    material, conf = predict_most_frequent_material(model_reflection, le_reflection, freq_start, freq_end)
    avg = avg_eff_reflection.get(material, 0)
    if avg < min_value:
        desc = f"Предсказанный материал — {material} (среднее {avg:.3f}) не удовлетворяет min={min_value}."
    else:
        desc = f"Предсказанный материал — {material} (среднее {avg:.3f}, уверенность {conf:.3f})."
    return {
        "1. Защитный внешний слой": "Полиуретановое покрытие",
        "2. Отражающий металлический слой": desc,
        "3. Импеданс-согласующий слой": "Пористая керамика",
        "4. Углеродный слой": "Углеродные нанотрубки",
        "5. Подложка из диэлектрика": "Полиимид"
    }

# Streamlit-интерфейс
st.title("Состав композитного материала")
criteria = st.selectbox(
    "Критерий оптимизации:",
    ["Эффективность экранирования", "Коэффициент поглощения", "Коэффициент отражения"]
)

freq_start = st.number_input("Начальная частота", value=1.0)
freq_end = st.number_input("Конечная частота", value=10.0)
min_val = st.number_input("Минимальное требуемое значение", value=1.0)

if st.button("Рассчитать"):
    if criteria == "Эффективность экранирования":
        comp = create_shielding_composite(freq_start, freq_end, min_val)
    elif criteria == "Коэффициент поглощения":
        comp = create_absorption_composite(freq_start, freq_end, min_val)
    else:
        comp = create_reflection_composite(freq_start, freq_end, min_val)

    st.subheader("Итоговый состав:")
    for layer, detail in comp.items():
        st.write(f"{layer}: {detail}")