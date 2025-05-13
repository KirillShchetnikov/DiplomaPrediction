import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def train_and_save(dataset_path, model_path, encoder_path, avg_eff_path):
    # Загружаем данные
    data = pd.read_csv(dataset_path)
    # Явное приведение типов
    data['frequency'] = pd.to_numeric(data['frequency'], errors='coerce')
    eff_col = data.columns[2]  # shielding_eff или absorption_eff или reflection_eff
    data[eff_col] = pd.to_numeric(data[eff_col], errors='coerce')
    data.dropna(subset=['frequency', eff_col], inplace=True)

    # Вход и метки
    X = data[['frequency', eff_col]].values
    le = LabelEncoder()
    y_enc = le.fit_transform(data['material'])
    num_classes = len(le.classes_)
    y_onehot = tf.keras.utils.to_categorical(y_enc, num_classes)

    # Среднее значение характеристики по материалу
    avg_eff = data.groupby('material')[eff_col].mean().to_dict()

    # Построение и обучение модели
    model = Sequential([
        Dense(10, activation='relu', input_shape=(2,)),
        Dense(10, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_onehot, epochs=100, verbose=1)

    # Сохранение модели, энкодера и среднего словаря
    model.save(model_path)
    joblib.dump(le, encoder_path)
    joblib.dump(avg_eff, avg_eff_path)
    print(f"Saved: {model_path}, {encoder_path}, {avg_eff_path}")


if __name__ == '__main__':
    # Эффективность экранирования
    train_and_save('dataset_ee.csv', 'model_ee.keras', 'le_ee.pkl', 'avg_eff_ee.pkl')
    # Коэффициент поглощения
    train_and_save('dataset_kp.csv', 'model_kp.keras', 'le_kp.pkl', 'avg_eff_kp.pkl')
    # Коэффициент отражения
    train_and_save('dataset_ko.csv', 'model_ko.keras', 'le_ko.pkl', 'avg_eff_ko.pkl')