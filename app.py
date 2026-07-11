import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Tampilan Halaman Web
st.set_page_config(page_title="Klasifikasi Batuan", page_icon="🪨")
st.title("Aplikasi Identifikasi Jenis Batuan Geologi 🪨")
st.write("Aplikasi ini dibuat untuk memenuhi tugas akhir/skripsi. Model yang digunakan dapat dipilih dari menu di samping.")

# 2. Menu Samping (Sidebar) untuk Pilih Model
st.sidebar.title("Pengaturan")
pilihan_model = st.sidebar.selectbox(
    "Pilih Arsitektur CNN:",
    ("DenseNet121", "ResNet50", "MobileNetV2")
)

# 3. Fungsi Memuat Model Berdasarkan Pilihan (Pakai Cache)
@st.cache_resource
def load_model(nama_model):
    if nama_model == "DenseNet121":
        return tf.keras.models.load_model('densenet121_model.keras')
    elif nama_model == "ResNet50":
        return tf.keras.models.load_model('resnet50_model.keras')
    else:
        return tf.keras.models.load_model('mobilenetv2_model.keras')

try:
    model = load_model(pilihan_model)
    st.sidebar.success(f"Model {pilihan_model} berhasil dimuat!")
except Exception as e:
    # Ini akan memunculkan tulisan error asli dari mesin Streamlit
    st.sidebar.error(f"Error aslinya adalah: {e}")

# 4. Daftar Kelas Batuan
class_names = ['Basalt', 'Coal', 'Granite', 'Limestone', 'Marble', 'Quartzite', 'Sandstone']

# 5. Tombol Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar batuan (jpg/png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 6. Tampilkan Gambar yang di-upload
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    st.write(f"Sedang menganalisis menggunakan **{pilihan_model}**...")
    
    # 7. Prapemrosesan Gambar (SUDAH DIPERBAIKI: Menggunakan Rescale 1./255)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Kunci perbaikan agar tebakan tidak nyangkut di Sandstone
    img_array = img_array / 255.0 
    
    # 8. Memprediksi Gambar
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    # 9. Menampilkan Hasil Akhir
    st.success(f"**Tebakan Model:** {class_names[predicted_class]}")
    st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")