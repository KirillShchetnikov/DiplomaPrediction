import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

##########################################
# Модель для критерия "Эффективность экранирования"
##########################################
st.write("Загрузка и обучение модели для Эффективности экранирования")
# Файл data.csv должен содержать столбцы: material, frequency, shielding_eff
data_shielding = pd.read_csv("data.csv")
data_shielding['frequency'] = pd.to_numeric(data_shielding['frequency'], errors='coerce')
data_shielding['shielding_eff'] = pd.to_numeric(data_shielding['shielding_eff'], errors='coerce')
data_shielding.dropna(subset=['frequency', 'shielding_eff'], inplace=True)

X_shielding = data_shielding['frequency'].values.reshape(-1, 1)
le_shielding = LabelEncoder()
y_shielding_encoded = le_shielding.fit_transform(data_shielding['material'])
num_classes_shielding = len(le_shielding.classes_)
y_shielding_onehot = tf.keras.utils.to_categorical(y_shielding_encoded, num_classes_shielding)
# Среднее значение эффективности по материалу (для проверки порога)
avg_eff_shielding = data_shielding.groupby('material')['shielding_eff'].mean().to_dict()

model_shielding = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(num_classes_shielding, activation='softmax')
])
model_shielding.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

st.write("Обучение модели для Эффективности экранирования...")
model_shielding.fit(X_shielding, y_shielding_onehot, epochs=100, verbose=0)
st.write("Обучение завершено для Эффективности экранирования.")


def predict_shielding_material(frequency):
    """Предсказывает материал для слоя экранирования по частоте."""
    pred = model_shielding.predict(np.array([[frequency]]))
    class_index = np.argmax(pred)
    material_name = le_shielding.inverse_transform([class_index])[0]
    confidence = pred[0][class_index]
    return material_name, confidence


# Фиксированные слои для шаблона Эффективности экранирования
external_protective_options = [
    "Полиуретановое покрытие"
]
dielectric_options = [
    "Полиимид с армирующей сеткой"
]
absorbing_inner_options = [
    "Графен"
]
internal_protective_options = [
    "Полиуретановое покрытие"
]


def create_shielding_composite(frequency, min_value):
    """Формирует состав композита для критерия Эффективности экранирования."""
    material, confidence = predict_shielding_material(frequency)
    avg_value = avg_eff_shielding.get(material, 0)

    if avg_value < min_value:
        layer2 = (f"Предсказанный материал — {material} "
                  f"(средняя эффективность {avg_value:.3f}) не удовлетворяет требованию (min = {min_value}).")
    else:
        layer2 = (f"Предсказанный материал — {material} "
                  f"(средняя эффективность {avg_value:.3f}, уверенность {confidence:.3f}).")

    composite = {
        "1. Защитный внешний слой": external_protective_options[0],
        "2. Экранирующий металлический слой": layer2,
        "3. Диэлектрический слой с армирующей сеткой": dielectric_options[0],
        "4. Поглощающий внутренний слой (опционально)": absorbing_inner_options[0],
        "5. Внутренний защитный слой": internal_protective_options[0],
    }
    return composite


##########################################
# Модель для критерия "Коэффициент поглощения"
##########################################
st.write("Загрузка и обучение модели для Коэффициента поглощения")
# Файл data_absorption.csv должен содержать столбцы: material, frequency, absorption_eff
data_absorption = pd.read_csv("data_absorption.csv")
data_absorption['frequency'] = pd.to_numeric(data_absorption['frequency'], errors='coerce')
data_absorption['absorption_eff'] = pd.to_numeric(data_absorption['absorption_eff'], errors='coerce')
data_absorption.dropna(subset=['frequency', 'absorption_eff'], inplace=True)

X_absorption = data_absorption['frequency'].values.reshape(-1, 1)
le_absorption = LabelEncoder()
y_absorption_encoded = le_absorption.fit_transform(data_absorption['material'])
num_classes_absorption = len(le_absorption.classes_)
y_absorption_onehot = tf.keras.utils.to_categorical(y_absorption_encoded, num_classes_absorption)
avg_eff_absorption = data_absorption.groupby('material')['absorption_eff'].mean().to_dict()

model_absorption = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(num_classes_absorption, activation='softmax')
])
model_absorption.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

st.write("Обучение модели для Коэффициента поглощения...")
model_absorption.fit(X_absorption, y_absorption_onehot, epochs=100, verbose=0)
st.write("Обучение завершено для Коэффициента поглощения.")


def predict_absorption_material(frequency):
    """Предсказывает материал для поглощающего композитного слоя по частоте."""
    pred = model_absorption.predict(np.array([[frequency]]))
    class_index = np.argmax(pred)
    material_name = le_absorption.inverse_transform([class_index])[0]
    confidence = pred[0][class_index]
    return material_name, confidence


# Фиксированные слои для шаблона Коэффициента поглощения
external_damping_options = [
    "Полиуретан"
]
magnetic_options = [
    "Никель-цинковый феррит"
]
polymeric_options = [
    "Графен"
]
structural_options = [
    "Стекловолокно"
]


def create_absorption_composite(frequency, min_value):
    """Формирует состав композита для критерия Коэффициента поглощения."""
    material, confidence = predict_absorption_material(frequency)
    avg_value = avg_eff_absorption.get(material, 0)

    if avg_value < min_value:
        layer3 = (f"Предсказанный материал для поглощающего композитного слоя — {material} "
                  f"(среднее значение {avg_value:.3f}) не удовлетворяет требованию (min = {min_value}).")
    else:
        layer3 = (f"Предсказанный материал для поглощающего композитного слоя — {material} "
                  f"(среднее значение {avg_value:.3f}, уверенность {confidence:.3f}).")

    composite = {
        "1. Внешний демпфирующий слой": external_damping_options[0],
        "2. Магнитный слой на основе феррита": magnetic_options[0],
        "3. Поглощающий композитный слой": layer3,
        "4. Полимерная прослойка (опционально для гибкости)": polymeric_options[0],
        "5. Внутренний структурный слой": structural_options[0],
    }
    return composite


##########################################
# Модель для критерия "Коэффициент отражения"
##########################################
st.write("Загрузка и обучение модели для Коэффициента отражения")
# Файл data_reflection.csv должен содержать столбцы: material, frequency, reflection_eff
data_reflection = pd.read_csv("data_reflection.csv")
data_reflection['frequency'] = pd.to_numeric(data_reflection['frequency'], errors='coerce')
data_reflection['reflection_eff'] = pd.to_numeric(data_reflection['reflection_eff'], errors='coerce')
data_reflection.dropna(subset=['frequency', 'reflection_eff'], inplace=True)

X_reflection = data_reflection['frequency'].values.reshape(-1, 1)
le_reflection = LabelEncoder()
y_reflection_encoded = le_reflection.fit_transform(data_reflection['material'])
num_classes_reflection = len(le_reflection.classes_)
y_reflection_onehot = tf.keras.utils.to_categorical(y_reflection_encoded, num_classes_reflection)
avg_eff_reflection = data_reflection.groupby('material')['reflection_eff'].mean().to_dict()

model_reflection = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(num_classes_reflection, activation='softmax')
])
model_reflection.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

st.write("Обучение модели для Коэффициента отражения...")
model_reflection.fit(X_reflection, y_reflection_onehot, epochs=100, verbose=0)
st.write("Обучение завершено для Коэффициента отражения.")


def predict_reflection_material(frequency):
    """Предсказывает материал для импеданс-согласующего слоя по частоте."""
    pred = model_reflection.predict(np.array([[frequency]]))
    class_index = np.argmax(pred)
    material_name = le_reflection.inverse_transform([class_index])[0]
    confidence = pred[0][class_index]
    return material_name, confidence


# Фиксированные слои для шаблона Коэффициента отражения
external_protective_options_reflection = [
    "Полиуретановое покрытие"
]
impedans_layer_options = [
    "Пористая керамика"
]
carbon_options = [
    "Углеродные нанотрубки"
]
dielectric_substrate_options = [
    "Полиимид"
]


def create_reflection_composite(frequency, min_value):
    """Формирует состав композита для критерия Коэффициента отражения."""
    material, confidence = predict_reflection_material(frequency)
    avg_value = avg_eff_reflection.get(material, 0)

    if avg_value < min_value:
        layer2 = (f"Предсказанный материал для импеданс-согласующего слоя — {material} "
                  f"(среднее значение {avg_value:.3f}) не удовлетворяет требованию (min = {min_value}).")
    else:
        layer2 = (f"Предсказанный материал для импеданс-согласующего слоя — {material} "
                  f"(среднее значение {avg_value:.3f}, уверенность {confidence:.3f}).")

    composite = {
        "1. Защитный внешний слой": external_protective_options_reflection[0],
        "2. Отражающий металлический слой": layer2,
        "3. Импеданс-согласующий слой": impedans_layer_options[0],
        "4. Углеродный слой": carbon_options[0],
        "5. Подложка из диэлектрика": dielectric_substrate_options[0],
    }
    return composite


##########################################
# Интерфейс Streamlit
##########################################
st.title("Состав композитного материала")

select_criteria = st.selectbox(
    "Выберите критерий оптимизации композитного материала:",
    ["Эффективность экранирования", "Коэффициент поглощения", "Коэффициент отражения"]
)

frequency_input = st.number_input("Введите диапазон частот", value=1.0)
min_value_input = st.number_input("Введите минимальное требуемое значение", value=1.0)

if st.button("Рассчитать"):
    if select_criteria == "Эффективность экранирования":
        composite_material = create_shielding_composite(frequency_input, min_value_input)
    elif select_criteria == "Коэффициент поглощения":
        composite_material = create_absorption_composite(frequency_input, min_value_input)
    elif select_criteria == "Коэффициент отражения":
        composite_material = create_reflection_composite(frequency_input, min_value_input)

    st.subheader("Состав композитного материала:")
    for layer, details in composite_material.items():
        st.write(f"{layer}: {details}")
