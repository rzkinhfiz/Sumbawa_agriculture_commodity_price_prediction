import streamlit as st

st.header('Tentang Aplikasi')
st.markdown(
    'Aplikasi ini membantu stakeholder pertanian di Kabupaten Sumbawa memantau harga historis dan melihat prediksi harga ke depan untuk komoditas utama.'
)

st.markdown('**Fitur utama:**')
st.markdown(
    '- Pilihan komoditas: Beras Premium, Beras Medium, Jagung Pipil Kering, dan Kacang Hijau.\n'
    '- Pilihan model: Univariate dan Multivariate.\n'
    '- Grafik historis harga dan forecast dengan rentang prediksi.\n'
    '- Metrik evaluasi model dan rekomendasi otomatis.\n'
    '- Desain glassmorphism untuk tampilan modern.'
)

st.subheader('Sumber Data')
st.markdown(
    '- Data harga bersih: `data/processed/price_cleaned.csv`\n'
    '- Data produktivitas: `data/processed/production_transformed.csv`\n'
    '- Model checkpoint terbaik: `models/best/`'
)

st.subheader('Panduan Penggunaan')
st.markdown(
    '1. Pilih komoditas.\n'  
    '2. Pilih jenis model.\n'  
    '3. Pilih horizon prediksi atau input custom.\n'  
    '4. Tambahkan produksi tambahan untuk melihat dampak pada prediksi.'  
)
