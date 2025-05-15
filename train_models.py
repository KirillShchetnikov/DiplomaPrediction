import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.cluster import KMeans
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
import os

def train_and_save_clustered(dataset_path, out_dir, n_clusters=3):
    # Создаем директорию для моделей
    os.makedirs(out_dir, exist_ok=True)

    # Загружаем данные
    data = pd.read_csv(dataset_path)
    data['frequency'] = pd.to_numeric(data['frequency'], errors='coerce')
    eff_col = data.columns[2]
    data[eff_col] = pd.to_numeric(data[eff_col], errors='coerce')
    data.dropna(subset=['frequency', eff_col], inplace=True)

    # Логарифмируем эффективность
    data[eff_col] = np.log1p(data[eff_col])

    # Группировка материалов по средней эффективности и кластеризация
    avg_eff_df = data.groupby('material')[eff_col].mean().reset_index()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    avg_eff_df['cluster'] = kmeans.fit_predict(avg_eff_df[[eff_col]])
    cluster_map = dict(zip(avg_eff_df['material'], avg_eff_df['cluster']))
    data['cluster'] = data['material'].map(cluster_map)

    # Сохраняем модель кластеризации
    joblib.dump(kmeans, os.path.join(out_dir, 'kmeans_cluster_model.pkl'))

    # Обучаем классификатор кластера
    cluster_scaler = QuantileTransformer(output_distribution='normal', random_state=0)
    X_cluster = cluster_scaler.fit_transform(data[['frequency', eff_col]])
    y_cluster = data['cluster'].values

    cluster_clf = Sequential([
        Input(shape=(2,)),
        Dense(32, activation='relu'),
        Dense(n_clusters, activation='softmax')
    ])
    cluster_clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cluster_clf.fit(X_cluster, y_cluster, epochs=100, verbose=1)

    cluster_clf.save(os.path.join(out_dir, 'cluster_classifier.keras'))
    joblib.dump(cluster_scaler, os.path.join(out_dir, 'cluster_scaler.pkl'))

    # Обучаем модели по каждому кластеру
    for cluster_id in sorted(data['cluster'].unique()):
        sub = data[data['cluster'] == cluster_id]

        # Преобразование данных
        scaler = QuantileTransformer(output_distribution='normal', random_state=0)
        X = scaler.fit_transform(sub[['frequency', eff_col]].values)
        le = LabelEncoder()
        y_enc = le.fit_transform(sub['material'])
        y_onehot = tf.keras.utils.to_categorical(y_enc)

        # Средние значения
        avg_eff = sub.groupby('material')[eff_col].mean().to_dict()

        # Создание модели
        model = Sequential([
            Input(shape=(2,)),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            Dropout(0.2),
            Dense(64),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            Dropout(0.2),
            Dense(len(le.classes_), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X, y_onehot, epochs=200, verbose=1)

        # Сохранение всех файлов
        model.save(os.path.join(out_dir, f'model_cluster_{cluster_id}.keras'))
        joblib.dump(le, os.path.join(out_dir, f'label_encoder_cluster_{cluster_id}.pkl'))
        joblib.dump(scaler, os.path.join(out_dir, f'scaler_cluster_{cluster_id}.pkl'))
        joblib.dump(avg_eff, os.path.join(out_dir, f'avg_eff_cluster_{cluster_id}.pkl'))
        print(f"Saved cluster {cluster_id} model and assets.")

if __name__ == '__main__':
    # train_and_save_clustered('dataset_ee.csv', out_dir='models_ee', n_clusters=3)
    # train_and_save_clustered('dataset_kp.csv', out_dir='models_kp', n_clusters=3)
     train_and_save_clustered('dataset_ko.csv', out_dir='models_ko', n_clusters=3)