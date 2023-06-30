import streamlit as st
from  PIL import Image
import inference_realesrgan as ir

st.set_page_config(
   page_title="Image Enhancer",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

st.header('Image Optimization Using Real-ESRGAN')
upload_img = st.file_uploader(label="Upload Your Image",type=['jpg','png','jpeg'])

col1,col2 = st.columns(2)

if upload_img is not None:
    image = Image.open(upload_img)

    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
        st.image(image,width=300)

    with col2:
        with st.spinner('Processing...Seat Back And Relax.'):
            st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True)
            result = ir.main(input='C:\\Users\\SUMIT\\Desktop\\IO\\Real-ESRGAN\\inputs\\image-31.png', outscale=3.5,fp32= '--fp32',  face_enhance='True',  ext='auto',output= 'results')

        st.image(result,width=300)
