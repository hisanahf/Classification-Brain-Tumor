import os

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from fpdf import FPDF
import tempfile

# CONFIG 
IMG_SIZE = 128
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# CSS
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

bg_img = get_base64_image("Beranda.png")

st.markdown(f"""
<style>

/* ===== BACKGROUND ===== */

.stApp {{
    background-image: url("data:image/png;base64,{bg_img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

.block-container {{
    padding-top: 0rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}}

/* Hilangkan warna header streamlit */

[data-testid="stHeader"] {{
    background: transparent;
}}

/* ===== FILE UPLOADER ===== */

[data-testid="stFileUploader"] {{
    width: 380px;
    margin-left: -80px;
}}

[data-testid="stFileUploader"] section {{
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 2px dashed white;
}}

/* ===== BUTTON ===== */

.stButton > button {{
    width: 220px;
    height: 50px;
    border-radius: 15px;

    background: linear-gradient(
        90deg,
        #00c6ff,
        #0072ff
    );

    color: white;
    font-weight: bold;
    border: none;
}}

/* ===== CARD HASIL ===== */

.result-card {{
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 20px;
    margin-top: 20px;
}}

</style>
""", unsafe_allow_html=True)

# MODEL
@st.cache_resource
def load_models():

    cls_model = tf.keras.models.load_model(
        "best_mobilenetv3.h5",
        compile=False
    )

    seg_model = tf.keras.models.load_model(
        "model_unet.h5",
        compile=False
    )

    return cls_model, seg_model


cls_model, seg_model = load_models()

# PREPROCESS MODEL
def preprocess_cls(image):

    img = image.resize((200, 200)).convert("RGB")

    img = np.array(img)

    img = mobilenet_preprocess(img)

    return np.expand_dims(img, axis=0)


def preprocess_seg(image):

    img = image.convert("L")

    img = np.array(img)

    # CLAHE Enhancement
    clahe = cv2.createCLAHE(2.0, (8, 8))

    img = clahe.apply(img)

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    ) / 255.0

    return np.expand_dims(img, (0, -1)), img


# PREDICT
def predict_all(image):

    # KLASIFIKASI
    pred_cls = cls_model.predict(
        preprocess_cls(image),
        verbose=0
    )

    class_id = np.argmax(pred_cls)

    label = CLASS_NAMES[class_id]

    # confidence
    confidence = float(pred_cls[0][class_id])

    # SEGMENTASI
    seg_input, img_gray = preprocess_seg(image)

    pred_seg = seg_model.predict(
        seg_input,
        verbose=0
    )[0]

    # jika classifier yakin no tumor
    # maka segmentasi dihapus

    if label == "notumor" and confidence > 0.90:

        mask = np.zeros_like(
            pred_seg,
            dtype=np.uint8
        )

    else:

        # threshold lebih ketat
        mask = (pred_seg > 0.8).astype(np.uint8)

        # hapus dimensi channel
        mask = np.squeeze(mask)

        # HAPUS NOISE
        area_pixel = np.sum(mask)

        if area_pixel < 500:

            mask[:] = 0


    # OVERLAY
    img_color = cv2.cvtColor(
        (img_gray * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    mask_resized = cv2.resize(
        mask.astype(np.uint8),
        (img_color.shape[1], img_color.shape[0])
    )

    # warna merah untuk tumor
    img_color[mask_resized == 1] = [255, 0, 0]

    return label, pred_cls[0], img_color


# BAGIAN PDF
def generate_pdf(image, seg_img, label, probs):
    pdf = FPDF()
    pdf.add_page()
    
    # HEADER
    pdf.set_font("Arial", style="B", size=18)
    pdf.set_text_color(26, 54, 93)  
    pdf.cell(190, 10, "HASIL SCREENING AWAL BRAIN TUMOR", ln=True, align="C")
    
    pdf.set_font("Arial", style="I", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(190, 5, "Sistem Analisis Citra MRI Berbasis Deep Learning", ln=True, align="C")
    
    # Garis Pembatas Header
    pdf.set_draw_color(26, 54, 93)
    pdf.set_line_width(0.8)
    pdf.line(10, 28, 200, 28)
    pdf.ln(12)
    
    # SIMPAN GAMBAR SEMENTARA
    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(tmp_input.name)
    
    tmp_seg = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(seg_img).save(tmp_seg.name)
    
    # ISI HASIL PDF
    pdf.set_font("Arial", style="B", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(100, 8, "Hasil Klasifikasi:", ln=False)
    pdf.cell(90, 8, "Nilai Probabilitas Detail:", ln=True)
    
    # Hasil Utama Bold Besar
    pdf.set_font("Arial", style="B", size=16)
    if label.lower() == "glioma":
        pdf.set_text_color(220, 53, 69) # Merah (Ganas)
    elif label.lower() == "notumor":
        pdf.set_text_color(40, 167, 69) # Hijau (Normal)
    else:
        pdf.set_text_color(255, 159, 64) # Oranye/Kuning (Jinak)
        
    pdf.cell(100, 12, f"TERINDIKASI: {label.upper()}", ln=False)
    
    # List Probabilitas Semua Kelas (Sisi Kanan)
    pdf.set_font("Arial", size=9)
    pdf.set_text_color(50, 50, 50)
    
    # Urutkan visualisasi probabilitas ke kanan
    y_current = pdf.get_y()
    for cls, p in zip(CLASS_NAMES, probs):
        pdf.set_x(110)
        pdf.cell(90, 6, f"- {cls.capitalize()}: {p*100:.2f}%", ln=True)
    
    pdf.set_y(y_current + 15) # Reset koordinat Y setelah kolom probabilitas selesai
    pdf.ln(5)
    
    # VISUALISASI CITRA MRI
    pdf.set_font("Arial", style="B", size=12)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(90, 8, "Citra Input MRI Asli", ln=False, align="C")
    pdf.cell(10, 8, "", ln=False)
    pdf.cell(90, 8, "Hasil Segmentasi", ln=True, align="C")
    
    # Peletakan Gambar Sejajar Horizontal
    y_img = pdf.get_y()
    pdf.image(tmp_input.name, x=15, y=y_img, w=80)
    pdf.image(tmp_seg.name, x=115, y=y_img, w=80)
    
    # Geser Y kebawah area gambar (Ukuran tinggi gambar estimasi 80mm)
    pdf.set_y(y_img + 82)
    pdf.ln(5)
    
    # CATATAN 
    pdf.set_draw_color(220, 220, 220)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()) # Garis batas tipis atas catatan
    pdf.ln(4)
    
    pdf.set_font("Arial", style="B", size=11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(190, 6, "Catatan:", ln=True)
    
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(60, 60, 60)
    
    # Informasi Deskriptif Berdasarkan Label
    if label.lower() != "notumor":
        pdf.multi_cell(190, 5, "1. Visualisasi warna merah pada citra merupakan Hasil Segmentasi yang menunjukkan koordinat area jaringan yang dicurigai sebagai tumor.")
        
    if label.lower() == "glioma":
        pdf.multi_cell(190, 5, "2. Tumor Glioma terdeteksi. Glioma memiliki umumnya bersifat malignant (ganas).")
    elif label.lower() == "meningioma":
        pdf.multi_cell(190, 5, "2. Tumor Meningioma terdeteksi. Jenis tumor ini umumnya tumbuh pada membran pelindung otak dan mayoritas bersifat jinak (benign).")
    elif label.lower() == "pituitary":
        pdf.multi_cell(190, 5, "2. Tumor Pituitary terdeteksi. Kategori tumor ini biasanya terbentuk pada kelenjar di bawah otak dan bersifat jinak (benign).")
    else:
        pdf.multi_cell(190, 5, "1. Berdasarkan analisis komputasi arsitektur deep learning, struktur jaringan otak pada citra MRI saat ini tidak memperlihatkan adanya tanda-tanda tumor yang signifikan.")
    
    pdf.ln(10)
    
    # DISCLAIMER 
    pdf.set_fill_color(245, 247, 250)
    pdf.set_text_color(120, 30, 30)
    pdf.set_font("Arial", style="B", size=9)
    pdf.cell(190, 5, "PENTING:", ln=True, fill=True)
    
    pdf.set_font("Arial", style="I", size=8.5)
    pdf.set_text_color(100, 100, 100)
    disclaimer_text = (
        "Dokumen ini diproduksi secara otomatis oleh sistem kecerdasan buatan "
        "dan murni ditujukan sebagai alat bantu screening awal saja. Hasil prediksi ini BUKAN "
        "merupakan diagnosis final medis."
    )
    pdf.multi_cell(190, 4, disclaimer_text, fill=True)
    
    file_path = "report.pdf"
    pdf.output(file_path)
    
    # Hapus manual file temporary agar ram server hemat
    try:
        os.unlink(tmp_input.name)
        os.unlink(tmp_seg.name)
    except:
        pass
        
    return file_path

# ================= LANDING SECTION =================

col1, col2 = st.columns([1.1, 1.5])

with col1:

    st.markdown(
        """
        <div style="margin-left:-80px; margin-top:80px;">

        <h1 style="
            color:white;
            font-size:48px;
            font-weight:700;
            line-height:1.1;
            margin-bottom:10px;
        ">
            Brain Tumor Detection
        </h1>

        <p style="
            color:white;
            font-size:20px;
        ">
            AI System MRI analysis for brain tumor.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "png", "jpeg"]
    )

with col2:
    st.empty()

# ================= HASIL ANALISIS =================

if uploaded_file:

    image = Image.open(uploaded_file)

    col_img, col_space = st.columns([1, 1.5])

    with col_img:

        st.image(
            image,
            caption="Input MRI",
            width=300
        )

        analisis = st.button("🔍 Analisis Sekarang")

        if analisis:

            with st.spinner("AI sedang bekerja..."):

                label, probs, seg_img = predict_all(image)

            st.subheader(f"Hasil: {label.upper()}")

            for cls, p in zip(CLASS_NAMES, probs):

                st.write(f"{cls}: {p*100:.2f}%")

                st.progress(float(p))

            if label.lower() != "notumor":

                st.image(
                    seg_img,
                    caption="Segmentasi Tumor (Merah)"
                )

                st.info(
                    "Catatan: Segmentasi warna merah digunakan untuk visualisasi area yang dicurigai sebagai tumor."
                )

            if label.lower() == "glioma":
                st.warning(
                    "Informasi Medis: Terindikasi adanya Tumor Glioma, tumor ini berpotensi ganas (malignant)."
                )

            elif label.lower() == "meningioma":
                st.success(
                    "Informasi Medis: Terindikasi adanya Tumor Meningioma, tumor ini umumnya merupakan tumor jinak (benign)."
        )

            elif label.lower() == "pituitary":
                st.success(
                    "Informasi Medis: Terindikasi adanya Tumor Pituitary, tumor ini umumnya merupakan tumor jinak (benign)."
        )

            else:

                st.success(
                    "Tidak ditemukan tumor"
        )

            pdf_file = generate_pdf(
                image,
                seg_img,
                label,
                probs
            )

            with open(pdf_file, "rb") as f:

                st.download_button(
                    "📄 Download Report",
                    f,
                    file_name="Hasil Screening Awal.pdf"
                )
    st.markdown(
    "</div>",
    unsafe_allow_html=True
    )