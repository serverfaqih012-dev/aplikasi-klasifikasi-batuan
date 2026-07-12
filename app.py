import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Tampilan Halaman Web
st.set_page_config(page_title="Klasifikasi Batuan", page_icon="🪨")
st.title("Aplikasi Identifikasi Jenis Batuan Geologi 🪨")
st.write("Aplikasi ini membandingkan 3 arsitektur CNN untuk klasifikasi batuan.")

# 2. Menu Samping (Sidebar)
st.sidebar.title("Pengaturan")
pilihan_model = st.sidebar.selectbox(
    "Pilih Arsitektur CNN:",
    ("DenseNet121", "ResNet50", "MobileNetV2")
)

# 3. Fungsi Memuat Model
@st.cache_resource(max_entries=1)
def load_model(nama_model):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    if nama_model == "DenseNet121":
        model_path = os.path.join(BASE_DIR, 'densenet121_model.keras')
    elif nama_model == "ResNet50":
        model_path = os.path.join(BASE_DIR, 'resnet50_model.keras')
    else:
        model_path = os.path.join(BASE_DIR, 'mobilenetv2_model.keras')
        
    return tf.keras.models.load_model(model_path)

try:
    model = load_model(pilihan_model)
    st.sidebar.success(f"Model {pilihan_model} siap digunakan!")
except Exception as e:
    st.sidebar.error(f"Terdapat Masalah: {e}")
    st.stop()

# 4. Daftar Kelas Batuan (Sudah Sesuai Urutan Colab)
class_names = ['Basalt', 'Coal', 'Granite', 'Limestone', 'Marble', 'Quartzite', 'Sandstone']

# 5. Tombol Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar batuan (jpg/png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    st.write(f"Sedang menganalisis dengan **{pilihan_model}**...")
    
    # 6. Prapemrosesan Gambar
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)  # Pastikan tipe data float
    img_array = np.expand_dims(img_array, axis=0)
    
    # KUNCI JAWABAN: Prapemrosesan otomatis mengikuti model yang dipilih (Sama dengan Colab)
    if pilihan_model == "DenseNet121":
        img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    elif pilihan_model == "ResNet50":
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    elif pilihan_model == "MobileNetV2":
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # 7. Memprediksi Gambar
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    # 8. Menampilkan Hasil Akhir
    st.success(f"**Tebakan Model:** {class_names[predicted_class]}")
    st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")