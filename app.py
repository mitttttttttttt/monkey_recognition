# 以下を「app.py」に書き込み
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("サル画像認識アプリ")
st.sidebar.write("オリジナルの画像認識モデルを使ってサルの種類を判定します。また、人の写真からどのサルに似ているか判定することも可能です。")

st.sidebar.write("")

img = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        # 予測
        results = predict(img)

        # 結果の表示
        st.subheader("判定結果")
        n_top = 1 
        for result in results[:n_top]:
            st.write(str(round(result[2]*100, 2)) + "%の確率で" + result[0] + "です。")
