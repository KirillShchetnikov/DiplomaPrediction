import numpy as np
import streamlit as st
import joblib
import tensorflow as tf

# Загрузка предобученных моделей и связанных артефактов
model_shielding = tf.keras.models.load_model('model_ee.keras')
le_shielding = joblib.load('le_ee.pkl')
avg_eff_shielding = joblib.load('avg_eff_ee.pkl')

model_absorption = tf.keras.models.load_model('model_kp.keras')
le_absorption = joblib.load('le_kp.pkl')
avg_eff_absorption = joblib.load('avg_eff_kp.pkl')

model_reflection = tf.keras.models.load_model('model_ko.keras')
le_reflection = joblib.load('le_ko.pkl')
avg_eff_reflection = joblib.load('avg_eff_ko.pkl')

# Функция универсального предсказания
def predict_material(model, le, frequency):
    pred = model.predict(np.array([[frequency]]))
    idx = np.argmax(pred)
    material = le.inverse_transform([idx])[0]
    confidence = pred[0][idx]
    return material, confidence

# Функции для формирования композита

def create_shielding_composite(frequency, min_value):
    material, conf = predict_material(model_shielding, le_shielding, frequency)
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


def create_absorption_composite(frequency, min_value):
    material, conf = predict_material(model_absorption, le_absorption, frequency)
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


def create_reflection_composite(frequency, min_value):
    material, conf = predict_material(model_reflection, le_reflection, frequency)
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
frequency = st.number_input("Частота", value=1.0)
min_val = st.number_input("Минимальное требуемое значение", value=1.0)

if st.button("Рассчитать"):
    if criteria == "Эффективность экранирования":
        comp = create_shielding_composite(frequency, min_val)
    elif criteria == "Коэффициент поглощения":
        comp = create_absorption_composite(frequency, min_val)
    else:
        comp = create_reflection_composite(frequency, min_val)

    st.subheader("Итоговый состав:")
    for layer, detail in comp.items():
        st.write(f"{layer}: {detail}")