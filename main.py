import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Загрузка данных из CSV-файла
# Предполагается, что CSV имеет столбцы: material, frequency, shielding_eff
# Пример строки: epoxy_resin,1.0000000000000,1.6883072408619
data = pd.read_csv("data.csv")
X = data['frequency'].values.reshape(-1, 1)
y = data['shielding_eff'].values.reshape(-1, 1)

# 2. Построение и обучение нейронной сети для предсказания параметра экранирующего слоя (слой 2)
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

st.write("Обучение нейронной сети...")
model.fit(X, y, epochs=100, verbose=0)
st.write("Обучение завершено.")


def predict_layer2_value(frequency):
    """
    Функция для предсказания параметра экранирующего металлического слоя
    на основе частоты с использованием обученной нейронной сети.
    """
    pred = model.predict(np.array([[frequency]]))
    return pred[0][0]


# 3. Определение вариантов материалов для остальных слоёв

# Слой 1: Защитный внешний слой (толщина 0.1–0.3 мм)
external_protective_options = [
    {"material": "Polyethylene", "thickness": "0.1–0.3 мм"},
    {"material": "Polypropylene", "thickness": "0.1–0.3 мм"}
]

# Слой 3: Диэлектрический слой с армирующей сеткой (толщина 0.5–1 мм)
dielectric_options = [
    {"material": "Glass fiber", "thickness": "0.5–1 мм"},
    {"material": "Aramid fiber", "thickness": "0.5–1 мм"}
]

# Слой 4: Поглощающий внутренний слой (опционально, толщина 0.3–0.5 мм)
absorbing_inner_options = [
    {"material": "Foam", "thickness": "0.3–0.5 мм"},
    {"material": "Cork", "thickness": "0.3–0.5 мм"}
]

# Слой 5: Внутренний защитный слой (толщина 0.1–0.3 мм)
internal_protective_options = [
    {"material": "Polyethylene", "thickness": "0.1–0.3 мм"},
    {"material": "Polypropylene", "thickness": "0.1–0.3 мм"}
]


def create_composite_material(frequency, min_efficiency):
    """
    Функция для формирования состава композитного материала.
    Для слоя 2 предсказывается значение нейронной сетью, остальные слои выбираются из списка.
    """
    # Предсказание для слоя 2
    predicted_value = predict_layer2_value(frequency)

    if predicted_value < min_efficiency:
        layer2_description = (f"Экранирующий металлический слой не удовлетворяет требованию "
                              f"(предсказанное значение: {predicted_value:.3f} < {min_efficiency}).")
    else:
        layer2_description = (f"Экранирующий металлический слой с предсказанным значением {predicted_value:.3f} "
                              f"(толщина: 0.05–0.1 мм).")

    # Формирование состава композитного материала
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

frequency = st.number_input("Введите частоту", value=1.0)
min_eff = st.number_input("Введите минимальную эффективность экранирования", value=1.0)

if st.button("Рассчитать"):
    composite_material = create_composite_material(frequency, min_eff)
    st.subheader("Состав композитного материала:")
    for layer, details in composite_material.items():
        st.write(f"{layer}: {details}")
