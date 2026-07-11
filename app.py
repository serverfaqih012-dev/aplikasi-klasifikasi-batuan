import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Tampilan Halaman Web
st.set_page_config(page_title="Klasifikasi Batuan", page_icon="🪨")
st.title("Aplikasi Identifikasi Jenis Batuan Geologi 🪨")
st.write("Aplikasi ini dibuat untuk memenuhi tugas akhir/skripsi. Model yang digunakan adalah DenseNet121 yang dilatih untuk mendeteksi 7 jenis batuan: Basalt, Coal, Granite, Limestone, Marble, Quartzite, dan Sandstone.")

# 2. Fungsi Memuat Model (Memori Cache agar web tidak lemot)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('densenet121_model.keras')
    return model

try:
    model = load_model()
except Exception as e:
    st.error("Gagal memuat model. Pastikan file densenet121_model.keras ada di folder yang sama!")

# 3. Daftar Kelas Batuan
class_names = ['Basalt', 'Coal', 'Granite', 'Limestone', 'Marble', 'Quartzite', 'Sandstone']

# 4. Tombol Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar batuan (jpg/png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 5. Tampilkan Gambar yang di-upload
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    st.write("Sedang menganalisis...")
    
    # 6. Prapemrosesan Gambar agar sesuai dengan input model
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    # INI YANG DIUBAH: Menyamakan format gambar dengan format di Google Colab
    img_array = img_array / 255.0
    
    # 7. Memprediksi Gambar
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    # 8. Menampilkan Hasil Akhir
    st.success(f"**Tebakan Model:** {class_names[predicted_class]}")
    st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")