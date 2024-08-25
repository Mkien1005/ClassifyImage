# -*- coding: utf-8 -*-
#pip install kaggle
#download dataset dogs-vs-cats thông qua câu lệnh kaggle competitions download -c dogs-vs-cats trên cmd
import os
import io
import joblib
import zipfile
import numpy as np
import pyarrow.hdfs
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from utils.hdfs_utils import save_model_to_hdfs
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

with zipfile.ZipFile('dogs-vs-cats.zip', 'r') as zip_ref:
    zip_ref.extractall('dogs_vs_cats')

# Đường dẫn đến tập tin train ZIP
zip_file_path = '/dogs_vs_cats/train.zip'
extract_to_path = '/dogs_vs_cats/'

# Giải nén tập tin
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)
    
# Đường dẫn đến tập tin test1 ZIP
zip_file_path = '/dogs_vs_cats/test1.zip'
extract_to_path = '/dogs_vs_cats'

# Giải nén tập tin
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

train_dir = '/content/dogs_vs_cats/train'

# Khởi tạo ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,        # Chuẩn hóa giá trị pixel từ [0, 255] thành [0, 1]
    shear_range=0.2,       # Làm nghiêng ảnh
    zoom_range=0.2,        # Mức độ zoom
    horizontal_flip=True   # Lật ảnh theo chiều ngang
)

# Tạo bộ dữ liệu huấn luyện từ thư mục đã làm sạch
train_generator = train_datagen.flow_from_directory(
    train_dir,       # Thư mục chứa ảnh đã làm sạch
    target_size=(150, 150),# Kích thước của ảnh đầu vào
    batch_size=32,         # Kích thước của mỗi batch
    class_mode='binary'    # Chế độ phân loại (nhị phân)
)
# Tạo mô hình Sequential
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10
)

# Lưu mô hình vào local
joblib.dump(model, 'model.pkl')

# Lưu mô hình vào HDFS
save_model_to_hdfs('model.pkl', '/user/student/model.pkl')

# Hiển thị đồ thị accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Hiển thị đồ thị loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

def predict(directory_path):
    results = {}
    for img_name in os.listdir(directory_path):
        img_path = os.path.join(directory_path, img_name)

        # Kiểm tra xem có phải là file ảnh không
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
            img_array /= 255.0  # Chia giá trị pixel cho 255

            prediction = model.predict(img_array)
            label = "Dog" if prediction[0] > 0.5 else "Cat"
            results[img_name] = label

    return results

# Sử dụng hàm để dự đoán các ảnh trong thư mục test1
directory_path = '/dogs_vs_cats/test1'
predictions = predict(directory_path)
# Đếm số lượng "Dog" và "Cat"
counter = Counter(predictions.values())
labels = ['Dogs', 'Cats']
counts = [counter['Dog'], counter['Cat']]

# Tạo biểu đồ tròn
plt.pie(counts, labels=labels, colors=['blue', 'orange'], autopct='%1.1f%%')
plt.title('Distribution of Images per Class')
plt.show()

# Kết nối đến HDFS
hdfs_client = pyarrow.hdfs.connect('localhost', 9000)

def read_image_from_hdfs(hdfs_path):
    # Mở tệp ảnh từ HDFS
    with hdfs_client.open(hdfs_path, 'rb') as f:
        img_bytes = f.read()
        img = Image.open(io.BytesIO(img_bytes))
        return img

def predict_from_hdfs_data(path):
    results = {}
    
    # Lấy danh sách các tệp trong thư mục HDFS
    for img_name in hdfs_client.ls(path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(path, img_name)
            img = read_image_from_hdfs(img_path)
            img = img.resize((150, 150))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
            img_array /= 255.0  # Chia giá trị pixel cho 255

            prediction = model.predict(img_array)
            label = "Dog" if prediction[0] > 0.5 else "Cat"
            results[img_name] = label

    return results

path = '/user/student/data'
predictions = predict_from_hdfs_data(path)
counter = Counter(predictions.values())
labels = ['Dogs', 'Cats']
counts = [counter['Dog'], counter['Cat']]

# Tạo biểu đồ tròn
plt.pie(counts, labels=labels, colors=['blue', 'orange'], autopct='%1.1f%%')
plt.title('Distribution of Images per Class')
plt.show()