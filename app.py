"""Dashboard Streamlit: Klasifikasi Citra Emosi Wajah (CNN).

Jalankan dari folder yang berisi app.py dan folder output/:
    streamlit run app.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

def find_out_dir():
    """Cari folder hasil training (berisi class_names.json), di mana pun posisinya di repo."""
    here = Path(__file__).parent
    for cand in (here / "output", here):
        if (cand / "class_names.json").exists():
            return cand
    for p in here.rglob("class_names.json"):
        return p.parent
    return here / "output"


OUT_DIR = find_out_dir()
IMG_SIZE = (48, 48)

st.set_page_config(page_title="Dashboard Emosi Wajah", page_icon="😊", layout="wide")


# ----------------------------------------------------------------- utilitas
@st.cache_data
def load_json(name):
    path = OUT_DIR / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def show_image(name, caption=None):
    path = OUT_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"File {name} tidak ditemukan di folder output/.")


@st.cache_resource(show_spinner="Memuat model CNN...")
def load_model():
    import tensorflow as tf

    path = OUT_DIR / "emotion_cnn_model.keras"
    if not path.exists():
        return None
    return tf.keras.models.load_model(path, compile=False)


def preprocess(img):
    """Sama dengan pipeline training: grayscale, resize 48x48, skala 0-255
    (normalisasi 0-1 dilakukan oleh layer Rescaling di dalam model)."""
    import tensorflow as tf

    arr = np.asarray(img.convert("L"), dtype=np.float32)[..., None]
    return tf.image.resize(arr, IMG_SIZE).numpy()


def crop_face(img):
    """Deteksi wajah terbesar dengan Haar cascade (OpenCV). Mengembalikan
    (gambar_crop, True) jika ketemu, atau (gambar_asli, False)."""
    try:
        import cv2
    except ImportError:
        return img, False
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        return img, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(0.1 * max(w, h))
    box = (max(x - m, 0), max(y - m, 0),
           min(x + w + m, img.width), min(y + h + m, img.height))
    return img.crop(box), True


def make_gradcam(batch, model, last_conv_name="last_conv"):
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(batch, training=False)
        pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]
    grads = tape.gradient(class_score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], IMG_SIZE).numpy().squeeze()
    return cam, int(pred_index)


def overlay(gray, cam, alpha=0.5):
    base = np.repeat(gray.squeeze()[..., None] / 255.0, 3, axis=-1)
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


# ------------------------------------------------------------------ halaman
def page_dataset():
    st.header("📊 Ringkasan Dataset")
    ds = load_json("dataset_summary.json")
    if ds is None:
        st.warning("dataset_summary.json tidak ditemukan. Jalankan notebook sampai Langkah 3.")
        return

    c = st.columns(5)
    c[0].metric("Total gambar", f"{ds['total_images']:,}")
    c[1].metric("Jumlah kelas", ds["num_classes"])
    c[2].metric("Training (80%)", f"{ds['train_images']:,}")
    c[3].metric("Validasi/Test (20%)", f"{ds['val_images']:,}")
    c[4].metric("Rasio imbalance", f"{ds['imbalance_ratio']:.2f}x")
    st.caption(f"Ukuran input {ds['image_size'][0]}x{ds['image_size'][1]} piksel, mode {ds['color_mode']}.")

    df = pd.DataFrame({
        "Training": ds["class_counts_train"],
        "Validasi": ds["class_counts_val"],
    })
    left, right = st.columns(2)
    with left:
        st.subheader("Jumlah gambar per kelas")
        st.bar_chart(df)
    with right:
        st.subheader("Tabel")
        df["Total"] = df["Training"] + df["Validasi"]
        df["Persen (%)"] = (df["Total"] / df["Total"].sum() * 100).round(2)
        st.dataframe(df, width="stretch")
    if ds["imbalance_ratio"] > 1.5:
        st.warning("Terdapat ketidakseimbangan data antar kelas (class imbalance).")

    st.subheader("Eksplorasi data")
    t1, t2, t3, t4 = st.tabs(["Distribusi kelas", "Contoh per kelas", "Kualitas gambar", "Contoh data training"])
    with t1:
        show_image("1_class_distribution.png")
    with t2:
        show_image("2_sample_per_class.png")
    with t3:
        show_image("3_image_quality.png")
    with t4:
        show_image("4_sample_training.png")


def page_training():
    st.header("🏋️ Hasil Training")
    hist = load_json("history.json")
    cfg = load_json("training_config.json")
    if hist is None:
        st.warning("history.json tidak ditemukan. Jalankan notebook sampai Langkah 4.")
        return

    best = int(np.argmax(hist["val_accuracy"])) + 1
    c = st.columns(4)
    c[0].metric("Epoch dijalankan", len(hist["loss"]))
    c[1].metric("Epoch terbaik", best)
    c[2].metric("Val accuracy terbaik", f"{max(hist['val_accuracy']) * 100:.2f}%")
    c[3].metric("Val loss (epoch terbaik)", f"{hist['val_loss'][best - 1]:.4f}")

    epochs = list(range(1, len(hist["loss"]) + 1))
    left, right = st.columns(2)
    with left:
        st.subheader("Accuracy")
        st.line_chart(pd.DataFrame(
            {"Train": hist["accuracy"], "Validation": hist["val_accuracy"]}, index=epochs))
    with right:
        st.subheader("Loss")
        st.line_chart(pd.DataFrame(
            {"Train": hist["loss"], "Validation": hist["val_loss"]}, index=epochs))

    if cfg:
        st.subheader("Konfigurasi training")
        st.table(pd.DataFrame({"Nilai": {k: str(v) for k, v in cfg.items()}}))


def page_evaluasi():
    st.header("✅ Evaluasi Model")
    ev = load_json("eval_metrics.json")
    cm_data = load_json("confusion_matrix.json")
    if ev is None:
        st.warning("eval_metrics.json tidak ditemukan. Jalankan notebook sampai Langkah 5.")
        return

    rep = ev["report"]
    names = ev["class_names"]
    c = st.columns(4)
    c[0].metric("Accuracy", f"{ev['accuracy'] * 100:.2f}%")
    c[1].metric("Precision (macro)", f"{rep['macro avg']['precision']:.3f}")
    c[2].metric("Recall (macro)", f"{rep['macro avg']['recall']:.3f}")
    c[3].metric("F1-score (macro)", f"{rep['macro avg']['f1-score']:.3f}")

    df = pd.DataFrame({k: v for k, v in rep.items() if isinstance(v, dict)}).T
    df = df.loc[names + ["macro avg", "weighted avg"]]
    df.columns = ["Precision", "Recall", "F1-Score", "Jumlah data"]
    df["Jumlah data"] = df["Jumlah data"].astype(int)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Metrik per kelas")
        st.dataframe(df.round(3), width="stretch")
        st.subheader("F1-score per kelas")
        st.bar_chart(df.loc[names, "F1-Score"])
        f1 = df.loc[names, "F1-Score"]
        st.write(f"Paling mudah dikenali: **{f1.idxmax()}** ({f1.max():.3f}). "
                 f"Paling sulit: **{f1.idxmin()}** ({f1.min():.3f}).")
    with right:
        st.subheader("Confusion matrix")
        if cm_data:
            cm = np.array(cm_data["matrix"])
            row = cm.sum(axis=1, keepdims=True)
            pct = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row != 0)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, label="Proporsi per kelas sebenarnya")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if pct[i, j] > 0.5 else "black", fontsize=8)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Kelas sebenarnya")
            plt.tight_layout()
            st.pyplot(fig)

            pairs = sorted(
                [(int(cm[i, j]), names[i], names[j])
                 for i in range(len(names)) for j in range(len(names)) if i != j],
                reverse=True)[:5]
            st.caption("Kesalahan terbanyak: " + "; ".join(
                f"{a} → {b} ({n})" for n, a, b in pairs))
        else:
            show_image("6_confusion_matrix.png")

    st.subheader("Contoh prediksi & Grad-CAM")
    t1, t2 = st.tabs(["Contoh prediksi data uji", "Grad-CAM"])
    with t1:
        show_image("7_sample_predictions.png")
    with t2:
        show_image("8_gradcam.png")


def page_prediksi():
    st.header("🔍 Prediksi Emosi dari Gambar")
    model = load_model()
    names = load_json("class_names.json")
    if model is None or names is None:
        st.warning("Model (emotion_cnn_model.keras) atau class_names.json tidak ditemukan di folder output/.")
        return

    st.write("Unggah foto wajah. Model dilatih pada wajah yang sudah di-crop, jadi hasil terbaik "
             "didapat dari foto wajah yang jelas dan menghadap depan.")
    auto_crop = st.checkbox("Deteksi dan crop wajah otomatis", value=True)
    src = st.radio("Sumber gambar", ["Unggah file", "Kamera"], horizontal=True)
    if src == "Unggah file":
        file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "bmp"])
    else:
        file = st.camera_input("Ambil foto")
    if file is None:
        return

    img = Image.open(file)
    note = ""
    if auto_crop:
        img, found = crop_face(img)
        note = "Wajah terdeteksi dan di-crop." if found else "Wajah tidak terdeteksi, memakai gambar utuh."

    x = preprocess(img)
    probs = model.predict(x[None, ...], verbose=0)[0]
    cam, pred = make_gradcam(x[None, ...], model)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img, caption=f"Gambar masukan. {note}", width="stretch")
    with c2:
        st.image(x.squeeze().astype("uint8"), caption="Setelah pra-pemrosesan (48x48, grayscale)",
                 clamp=True, width="stretch")
    with c3:
        st.image(overlay(x, cam), caption="Grad-CAM: area merah/kuning paling berpengaruh",
                 width="stretch")

    st.success(f"Prediksi: **{names[pred]}** (keyakinan {probs[pred] * 100:.1f}%)")
    st.subheader("Probabilitas per kelas")
    st.bar_chart(pd.Series(probs * 100, index=names, name="Probabilitas (%)"))


# --------------------------------------------------------------------- main
st.title("😊 Klasifikasi Citra Emosi Wajah dengan CNN")
if not (OUT_DIR / "class_names.json").exists():
    st.error("File hasil training (class_names.json, emotion_cnn_model.keras, dst.) tidak ditemukan di repo. "
             "Pastikan folder output/ dari Colab sudah di-upload ke GitHub.")
    here = Path(__file__).parent
    st.write("Isi folder aplikasi saat ini:")
    st.code("\n".join(sorted(p.name + ("/" if p.is_dir() else "") for p in here.iterdir()
                              if not p.name.startswith("."))))
    st.stop()

PAGES = {
    "Ringkasan Dataset": page_dataset,
    "Hasil Training": page_training,
    "Evaluasi Model": page_evaluasi,
    "Prediksi Gambar": page_prediksi,
}
choice = st.sidebar.radio("Menu", list(PAGES))
st.sidebar.caption("Data dibaca dari folder output/ hasil notebook.")
PAGES[choice]()
