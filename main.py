import numpy as np
import streamlit as st
import joblib
import tensorflow as tf
import os
from collections import Counter

# Определение папок для разных критериев
MODEL_DIRS = {
    "Эффективность экранирования": "models_ee",
    "Коэффициент поглощения": "models_kp",
    "Коэффициент отражения": "models_ko"
}

# Универсальная функция предсказания кластера и материала
def predict_most_frequent_material(model_dir, freq_start, freq_end, eff_value):
    # Загрузка кластерного классификатора и нормализатора
    cluster_model = tf.keras.models.load_model(os.path.join(model_dir, 'cluster_classifier.keras'))
    scaler_global = joblib.load(os.path.join(model_dir, 'cluster_scaler.pkl'))

    # Определение кластера
    cluster_inputs = np.array([[f, eff_value] for f in range(int(freq_start), int(freq_end) + 1)])
    cluster_inputs_scaled = scaler_global.transform(cluster_inputs)
    cluster_probs = cluster_model.predict(cluster_inputs_scaled)
    cluster_preds = np.argmax(cluster_probs, axis=1)
    most_common_cluster = Counter(cluster_preds).most_common(1)[0][0]

    # Загрузка соответствующей модели и энкодера
    model_path = os.path.join(model_dir, f'model_cluster_{most_common_cluster}.keras')
    le_path = os.path.join(model_dir, f'label_encoder_cluster_{most_common_cluster}.pkl')
    scaler_path = os.path.join(model_dir, f'scaler_cluster_{most_common_cluster}.pkl')
    avg_eff_path = os.path.join(model_dir, f'avg_eff_cluster_{most_common_cluster}.pkl')

    model = tf.keras.models.load_model(model_path)
    le = joblib.load(le_path)
    scaler = joblib.load(scaler_path)
    avg_eff = joblib.load(avg_eff_path)

    # Предсказание материала по диапазону частот
    X_input = np.array([[f, eff_value] for f in range(int(freq_start), int(freq_end) + 1)])
    X_input_scaled = scaler.transform(X_input)
    predictions = [np.argmax(model.predict(np.expand_dims(x, axis=0), verbose=0)) for x in X_input_scaled]
    most_common_idx = Counter(predictions).most_common(1)[0][0]
    material = le.inverse_transform([most_common_idx])[0]
    confidence = predictions.count(most_common_idx) / len(predictions)
    avg = avg_eff.get(material, 0)
    return material, confidence, avg

# Шаблоны слоёв
def create_composite(model_dir, freq_start, freq_end, min_value, base_layers):
    material, conf, avg = predict_most_frequent_material(model_dir, freq_start, freq_end, min_value)
    if avg < min_value:
        desc = f"Предсказанный материал — {material} (среднее {avg:.3f}) не удовлетворяет min={min_value}."
    else:
        desc = f"Предсказанный материал — {material} (среднее {avg:.3f}, уверенность {conf:.3f})."
    composite = base_layers.copy()
    for k in composite:
        if "слой" in k.lower() and "материал" in composite[k].lower():
            composite[k] = desc
    return composite

# Streamlit-интерфейс
st.title("Состав композитного материала")
criteria = st.selectbox(
    "Критерий оптимизации:",
    ["Эффективность экранирования", "Коэффициент поглощения", "Коэффициент отражения"]
)

freq_start = st.number_input("Начальная частота (ГГц)", value=1.0)
freq_end = st.number_input("Конечная частота (ГГц)", value=10.0)
min_val = st.number_input("Минимальное требуемое значение эффективности", value=1.0)

if st.button("Рассчитать"):
    model_dir = MODEL_DIRS[criteria]

    if criteria == "Эффективность экранирования":
        base_layers = {
            "1. Защитный внешний слой": "Полиуретановое покрытие",
            "2. Экранирующий металлический слой": "Материал рассчитывается",
            "3. Диэлектрический слой с армирующей сеткой": "Полиимид с армирующей сеткой",
            "4. Поглощающий внутренний слой (опционально)": "Графен",
            "5. Внутренний защитный слой": "Полиуретановое покрытие"
        }

    elif criteria == "Коэффициент поглощения":
        base_layers = {
            "1. Внешний демпфирующий слой": "Полиуретан",
            "2. Магнитный слой на основе феррита": "Никель-цинковый феррит",
            "3. Поглощающий композитный слой": "Материал рассчитывается",
            "4. Полимерная прослойка (опционально для гибкости)": "Графен",
            "5. Внутренний структурный слой": "Стекловолокно"
        }

    else:
        base_layers = {
            "1. Защитный внешний слой": "Полиуретановое покрытие",
            "2. Отражающий металлический слой": "Материал рассчитывается",
            "3. Импеданс-согласующий слой": "Пористая керамика",
            "4. Углеродный слой": "Углеродные нанотрубки",
            "5. Подложка из диэлектрика": "Полиимид"
        }

    comp = create_composite(model_dir, freq_start, freq_end, min_val, base_layers)

    st.subheader("Итоговый состав:")
    for layer, detail in comp.items():
        st.write(f"{layer}: {detail}")
