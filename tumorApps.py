import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Brain Scan - Classification",
    page_icon="brain.png",
    layout="wide"
)

def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_html(file_path):
    with open(file_path, "r") as f:
        st.markdown(f.read(), unsafe_allow_html=True)

load_css("style.css")

# ---------------------------------------------------------
# 2. DATASET INFO & MODEL LOAD
# ---------------------------------------------------------
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor (Sehat)', 'Pituitary']

TUMOR_INFO = {
    'Glioma': {
        'deskripsi': 'Glioma tumbuh pada sel glial di otak dan sering kali memiliki batas yang samar dengan jaringan sekitarnya.',
        'gejala': ['Sakit kepala berulang', 'Kejang-kejang', 'Penurunan memori', 'Kelemahan anggota tubuh']
    },
    'Meningioma': {
        'deskripsi': 'Meningioma tumbuh pada selaput meninges (ekstra-aksial) dengan batas tegas dan menyerap kontras secara merata.',
        'gejala': ['Pandangan kabur atau ganda', 'Sakit kepala bertahap', 'Gangguan pendengaran/penciuman']
    },
    'No Tumor (Sehat)': {
        'deskripsi': 'Jaringan otak berada dalam kondisi normal tanpa adanya indikasi massa tumor abnormal.',
        'gejala': ['Tidak ada indikasi gejala klinis tumor otak.']
    },
    'Pituitary': {
        'deskripsi': 'Tumor tumbuh pada kelenjar hipofisis di dasar otak yang mengatur keseimbangan hormon tubuh.',
        'gejala': ['Gangguan hormon', 'Kehilangan penglihatan samping', 'Kelelahan ekstrem']
    }
}

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load("2pt.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 3. SIDEBAR PANEL
# ---------------------------------------------------------
with st.sidebar:
   
    # Membungkus st.image dengan class CSS agar tetap bisa di-style
    st.markdown('<div class="sidebar-icon-container">', unsafe_allow_html=True)
    st.image("brain1.png", width=500) # Pastikan nama file sesuai
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.title("Panel Kontrol")
    uploaded_file = st.file_uploader("Unggah Scan MRI", type=["jpg", "png", "jpeg"])
    
    st.divider()
    st.markdown("### ⚙️ Informasi Model")
    st.info("Model: **EfficientNet-B0**\nWeights: `2pt.pth` (Masoud Dataset)")

# ---------------------------------------------------------
# 4. HALAMAN UTAMA & LOGIKA ANALISIS
# ---------------------------------------------------------
render_html("index.html")

if uploaded_file is None:
    st.markdown("""
        <div class="info-card" style="text-align: center; padding: 40px;">
            <h4 style="color: #555;">👈 Silakan unggah gambar MRI pada Sidebar untuk memulai diagnosa.</h4>
        </div>
    """, unsafe_allow_html=True)
else:
    col1, col2 = st.columns([1, 1.2], gap="large")
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.markdown("##### 🖼️ Citra MRI Input")
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("##### 📊 Analisis AI & Hasil Diagnosa")
        
        # Eksekusi Otomatis
        with st.spinner("Menganalisis matriks piksel MRI..."):
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_class = torch.max(probabilities, 0)
            
            label_hasil = CLASS_NAMES[predicted_class.item()]
            persentase = confidence.item() * 100
            probs_list = [p.item() * 100 for p in probabilities]

        # Tampilan Hasil
        st.metric(label="Prediksi Kelas", value=label_hasil, delta=f"{persentase:.1f}% Confidence")
        st.progress(int(persentase))
        
        if label_hasil == "No Tumor (Sehat)":
            st.success("✅ **Hasil Normal:** Jaringan otak teridentifikasi sehat.")
        else:
            st.error(f"⚠️ **Terindikasi:** Terdeteksi keberadaan **{label_hasil}**.")

        # Tabs Informasi Interaktif
        st.write("")
        tab1, tab2 = st.tabs(["📄 Informasi Penyakit", "📈 Probabilitas Detail"])
        
        with tab1:
            info = TUMOR_INFO[label_hasil]
            st.write(info['deskripsi'])
            st.markdown("**Gejala Umum:**")
            for g in info['gejala']:
                st.markdown(f"<div class='gejala-item'>• {g}</div>", unsafe_allow_html=True)
                
        with tab2:
            chart_data = pd.DataFrame({
                "Jenis Tumor": CLASS_NAMES,
                "Probabilitas (%)": probs_list
            })
            st.bar_chart(chart_data.set_index("Jenis Tumor"))

st.divider()
st.caption("⚠️ **Disclaimer:** Hasil ini merupakan prediksi model AI dan tidak menggantikan keputusan diagnosa resmi dokter radiologi.")