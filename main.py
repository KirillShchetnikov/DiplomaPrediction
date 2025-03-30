import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

# 1. Загрузка данных из CSV-файла
# Предполагается, что CSV имеет столбцы: material,frequency,shielding_eff
# Пример строки: epoxy_resin,1.0000000000000,1.6883072408619
data = pd.read_csv("data.csv")

# Приведение столбцов к числовому типу
data['frequency'] = pd.to_numeric(data['frequency'], errors='coerce')
data['shielding_eff'] = pd.to_numeric(data['shielding_eff'], errors='coerce')
data.dropna(subset=['frequency', 'shielding_eff'], inplace=True)

# Используем частоту как вход, а material как метку (target)
X = data['frequency'].values.reshape(-1, 1)

# Кодирование названий материала в числовые метки
le = LabelEncoder()
y_encoded = le.fit_transform(data['material'])
num_classes = len(le.classes_)
y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes)

# Для проверки эффективности материала рассчитываем среднюю shielding_eff для каждого материала
avg_eff = data.groupby('material')['shielding_eff'].mean().to_dict()

# 2. Построение и обучение нейронной сети для классификации материала (слой 2)
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

st.write("Обучение нейронной сети для предсказания материала 2-го слоя...")
model.fit(X, y_onehot, epochs=100, verbose=0)
st.write("Обучение завершено.")


def predict_layer2_material(frequency):
    """
    Функция для предсказания материала экранирующего слоя (слой 2) по частоте.
    Возвращает название материала и уверенность предсказания.
    """
    pred = model.predict(np.array([[frequency]]))
    class_index = np.argmax(pred)
    material_name = le.inverse_transform([class_index])[0]
    confidence = pred[0][class_index]
    return material_name, confidence


# 3. Определение вариантов для остальных слоёв (фиксированные варианты)
external_protective_options = [
    {"material": "Polyethylene", "thickness": "0.1–0.3 мм"},
    {"material": "Polypropylene", "thickness": "0.1–0.3 мм"}
]

dielectric_options = [
    {"material": "Glass fiber", "thickness": "0.5–1 мм"},
    {"material": "Aramid fiber", "thickness": "0.5–1 мм"}
]

absorbing_inner_options = [
    {"material": "Foam", "thickness": "0.3–0.5 мм"},
    {"material": "Cork", "thickness": "0.3–0.5 мм"}
]

internal_protective_options = [
    {"material": "Polyethylene", "thickness": "0.1–0.3 мм"},
    {"material": "Polypropylene", "thickness": "0.1–0.3 мм"}
]


def create_composite_material(frequency, min_efficiency):
    """
    Формирование состава композитного материала.
    Для слоя 2 предсказывается материал с помощью нейронной сети.
    Если средняя эффективность предсказанного материала ниже требуемой,
    выводится предупреждение.
    Остальные слои выбираются из фиксированного списка.
    """
    material, confidence = predict_layer2_material(frequency)
    material_avg_eff = avg_eff.get(material, 0)

    if material_avg_eff < min_efficiency:
        layer2_description = (f"Предсказанный материал для слоя 2 — {material} "
                              f"(средняя эффективность {material_avg_eff:.3f}) не удовлетворяет требованию "
                              f"(min_eff = {min_efficiency}).")
    else:
        layer2_description = (f"Предсказанный материал для слоя 2 — {material} "
                              f"(средняя эффективность {material_avg_eff:.3f}, уверенность {confidence:.3f}).")

    composite = {
        "1. Защитный внешний слой": external_protective_options[0],
        "2. Экранирующий металлический слой": layer2_description,
        "3. Диэлектрический слой с армирующей сеткой": dielectric_options[0],
        "4. Поглощающий внутренний слой (опционально)": absorbing_inner_options[0],
        "5. Внутренний защитный слой": internal_protective_options[0],
    }
    return composite


# 4. Реализация интерфейса через Streamlit
st.title("Состав композитного материала")

frequency_input = st.number_input("Введите частоту", value=1.0)
min_eff_input = st.number_input("Введите минимальную эффективность экранирования", value=1.0)

if st.button("Рассчитать"):
    composite_material = create_composite_material(frequency_input, min_eff_input)
    st.subheader("Состав композитного материала:")
    for layer, details in composite_material.items():
        st.write(f"{layer}: {details}")
