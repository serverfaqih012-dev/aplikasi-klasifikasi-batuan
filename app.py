import os
import streamlit as st

st.set_page_config(page_title="Detektif Error", page_icon="🕵️‍♂️")
st.title("🕵️‍♂️ Detektif File Streamlit")
st.write("Mari kita cek wujud asli file modelmu di dalam mesin Streamlit!")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models = ['densenet121_model.keras', 'resnet50_model.keras', 'mobilenetv2_model.keras']

for model_name in models:
    file_path = os.path.join(BASE_DIR, model_name)
    if os.path.exists(file_path):
        size_kb = os.path.getsize(file_path) / 1024
        size_mb = size_kb / 1024
        st.success(f"✅ **{model_name}** ADA di server!")
        st.write(f"➤ Ukuran terbaca di Streamlit: **{size_mb:.2f} MB**")
        
        if size_mb < 1:
            st.error("🚨 ERROR: Ukurannya terlalu kecil! File ini berubah jadi teks saat masuk GitHub (Git LFS), bukan file aslinya.")
    else:
        st.error(f"❌ **{model_name}** TIDAK DITEMUKAN di server!")