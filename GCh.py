import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle


# Функция для аугментации данных
def augment_data(images, labels):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    datagen.fit(images)
    return datagen


# Функция для обучения модели
def train_model(images, labels):
    # Перемешиваем данные перед разделением
    images, labels = shuffle(images, labels, random_state=42)

    base_model = MobileNetV2(include_top=False, input_shape=(96, 96, 3), weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Добавление Dropout слоя
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Разморозка некоторых верхних слоев для более точной настройки
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    datagen = augment_data(images, labels)

    # Разделение на обучающие и проверочные наборы
    train_gen = datagen.flow(images, labels, batch_size=32, subset='training')
    val_gen = datagen.flow(images, labels, batch_size=32, subset='validation')

    history = model.fit(train_gen, validation_data=val_gen, epochs=140)
    return model, history


# Функция для предсказания класса объекта на изображении
def predict_class(model, image):
    image_resized = cv2.resize(image, (96, 96))
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    image_normalized = np.array(image_resized).reshape(-1, 96, 96, 3) / 255.0
    result = model.predict(image_normalized, verbose=0)[0]
    return np.round(result)[0], result


# Загрузка обучающих данных
def load_training_data(directory):
    images = []
    labels = []
    if not os.path.isdir(directory):
        raise ValueError(f"Директория не существует: {directory}")


    for label, subdir in enumerate(['bicycle', 'not_bicycle']):
        subdir_path = os.path.join(directory, subdir)
        if not os.path.isdir(subdir_path):
            raise ValueError(f"Директория не существует: {subdir_path}")
        files = os.listdir(subdir_path)
        for idx, filename in enumerate(files):
            if filename.endswith(('.png', '.jpeg', '.jpg')):
                path = os.path.join(subdir_path, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (96, 96))
                    images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
                    labels.append(label)
                else:
                    print(f"Ошибка загрузки изображения: {path}")
            else:
                print(f"Пропуск файла (не .png, .jpeg, .jpg): {filename}")

    if not images or not labels:
        raise ValueError("Списки изображений или меток пусты. Проверьте директорию и файлы изображений.")

    images = np.array(images) / 255.0
    labels = np.array(labels)
    return images, labels



def display_bicycle_images(model, directory):
    files = os.listdir(directory)
    for filename in files:
        if filename.endswith(('.png', '.jpeg', '.jpg')):
            path = os.path.join(directory, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                prediction, probabilities = predict_class(model, img)
                if prediction == 0:  # Предположим, что 0 - это класс "велосипед"
                    print(f"Велосипед найден на изображении: {filename}")
                    plt.imshow(img, cmap='gray')
                    plt.title(f"Prediction: {int(prediction)}, Probability: {probabilities[0]:.4f}")
                    plt.axis('off')
                    plt.show()
                else:
                    print(f"Велосипед не найден на изображении: {filename}")
            else:
                print(f"Ошибка загрузки изображения: {path}")


# Основная часть кода
training_directory = '/Users/yaroslav/Documents/Python/Bicycle/dataset'
training_images, training_labels = load_training_data(training_directory)

# Обучение модели
model, history = train_model(training_images, training_labels)


# Визуализация обучения
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()


plot_training_history(history)

# Директория с тестовыми изображениями
test_directory = '/Users/yaroslav/Documents/Python/Bicycle/data/bikes'
if not os.path.isdir(test_directory):
    raise ValueError(f"Директория не существует: {test_directory}")

# Отображение изображений с велосипедами
display_bicycle_images(model, test_directory)
