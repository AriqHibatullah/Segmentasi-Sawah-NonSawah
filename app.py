import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from segmentation import preprocessing, hitung_sdmap, split, merge_blocks, remove_small_classes, combine, simplify_to_two_classes, overlay_mask

st.set_page_config(layout="wide")
st.title("Segmentasi Sawah vs Non-sawah")
st.write("Sistem segmentasi sawah dan non sawah menggunakan algoritma split and merge dan fitur sd map, warna, dan glcm.")

if "progress" not in st.session_state:
    st.session_state.progress = 0
if "gambar" not in st.session_state:
    st.session_state.gambar = None
if "last_uploaded_image" not in st.session_state:
    st.session_state.last_uploaded_image = None
if "preprocessing" not in st.session_state:
     st.session_state.preprocessing = False
if "canvas" not in st.session_state:
     st.session_state.canvas = None
if "sd_map" not in st.session_state:
     st.session_state.sd_map = None
if "blok" not in st.session_state:
     st.session_state.blok = None
if "sd_map" not in st.session_state:
     st.session_state.sd_map = None
if "class_map" not in st.session_state:
     st.session_state.class_map = None
if "cleaned_map" not in st.session_state:
     st.session_state.cleaned_map = None
if "filled_map" not in st.session_state:
     st.session_state.filled_map = None
if "binary_map" not in st.session_state:
     st.session_state.binary_map = None
if "overlayed" not in st.session_state:
     st.session_state.overlayed = None

st.caption("Progress segmentasi :")
progress_bar = st.progress(st.session_state.progress)

def resize_image(image, max_height):
    width, height = image.size
    scale = max_height / height
    new_height = max_height
    new_width = int(width * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Upload gambar", 
                                                                "Preprocessing", 
                                                                "SD Map", 
                                                                "Split", 
                                                                "Merge", 
                                                                "Eliminasi", 
                                                                "Region growing",
                                                                "Reclassification",
                                                                "Visualisasi"])

with tab1:
        image = st.file_uploader("Silahkan upload citra satelit anda untuk memulai segmentasi.", type=["jpg", "jpeg", "png"])
        if image != st.session_state.last_uploaded_image:
            st.session_state.gambar = image
            st.session_state.last_uploaded_image = image

        if st.session_state.gambar is not None:
            img = Image.open(image)
            img_resized = resize_image(img, max_height=300)
            st.image(img_resized, caption="Gambar yang diupload")

            if st.button("Preprocessing"):
                if st.session_state.progress < 12:
                        st.session_state.progress += 12
                        progress_bar.progress(st.session_state.progress)

                image_rgb = np.array(img)
                image_bgr = image_rgb[..., ::-1]
                canvas, image_rgb, lab, gray = preprocessing(image_bgr)

                st.session_state.canvas = canvas
                st.session_state.image_rgb = image_rgb
                st.session_state.lab = lab
                st.session_state.gray = gray

                if st.session_state.canvas is not None:
                    st.session_state.preprocessing = True
                    st.write("Gambar anda telah dilakukan preprocessing! silahkan periksa di tab selanjutnya.")

with tab2:
    if st.session_state.preprocessing is True:
        if st.session_state.canvas is not None:
            st.write("Citra asli dilakukan proses grayscaled, Gaussian blur, dan padding jika ukuran citra tidak kuadrat.")
            col1, col2, col3 = st.columns([2,2,3])

            with col1:
                plt.figure(figsize=(9, 5))

                plt.subplot(1, 1, 1)
                plt.imshow(st.session_state.image_rgb)
                plt.title('Citra Asli')
                plt.axis('off')
                st.pyplot(plt)
                st.caption(f"Ukuran canvas: {st.session_state.canvas.shape}")
            with col2:
                plt.figure(figsize=(9, 5))

                plt.subplot(1, 1, 1)
                plt.imshow(st.session_state.canvas, cmap = "gray")
                plt.title('Canvas')
                plt.axis('off')
                st.pyplot(plt)

            button1, button2 , pad = st.columns([3,3,5])
            
            with button1:
                R = st.number_input("Ukuran kernel untuk menghitung SD Map:", min_value=1, max_value=50, value=3, step=1)
            with button2:
                 st.write(" ")
                 with st.expander("ℹ️ Panduan ukuran kernel"):
                    st.markdown("""
                    - **Tidak aktif**: Jarang olahraga  
                    - **Sedikit aktif**: Olahraga 1-3 kali/minggu  
                    - **Cukup aktif**: Olahraga 3-5 kali/minggu  
                    - **Aktif**: Olahraga 6-7 kali/minggu  
                    - **Sangat aktif**: Olahraga 2 kali sehari  
                    """)

            if st.button("Buat SD Map"):
                if st.session_state.progress >= 12 and st.session_state.progress < 25:
                    st.session_state.progress += 12
                    progress_bar.progress(st.session_state.progress)

                sd_map, sd_map_norm = hitung_sdmap(st.session_state.canvas, R)
                st.session_state.sd_map = sd_map
                st.session_state.sd_map_norm = sd_map_norm
                if st.session_state.sd_map is not None:
                    st.write("Standard deviasi map sudah dihitung! silahkan periksa di tab selanjutnya.")
    else:
        st.session_state.preprocessing = None

with tab3:
    if st.session_state.sd_map is not None:
        st.write("Telah diperoleh Standard Deviation map.")

        col1, col2, col3 = st.columns([2,2,3])

        with col1:
            plt.figure(figsize=(9, 5))

            plt.subplot(1, 1, 1)
            plt.imshow(st.session_state.sd_map_norm, cmap="gray")
            plt.title('Standard Deviation Map')
            plt.axis('off')
            st.pyplot(plt)

        with col2:
            min_size = st.selectbox("Ukuran miniman blok hasil split:", [1, 2, 4, 8, 16, 32, 64, 128, 256])
            p_split = st.number_input("p:", min_value=0.1, max_value=0.9, value=0.82, step=0.1)
            sigma_split = np.percentile(st.session_state.sd_map[st.session_state.sd_map >= 0], p_split * 100)
            split_threshold = sigma_split ** 2

            if st.button("Lakukan split"):
                if st.session_state.progress >= 24 and st.session_state.progress < 37:
                    st.session_state.progress += 12
                    progress_bar.progress(st.session_state.progress)
                
                blok = split(st.session_state.canvas, 0, 0, st.session_state.canvas.shape[0], st.session_state.canvas.shape[1], min_size, split_threshold, depth=0)
                st.session_state.blok = blok
                if st.session_state.blok is not None:
                    st.write("gambar sudah di split! silahkan periksa di tab selanjutnya.")
    else:
            st.session_state.sd_map = None

with tab4:
    if st.session_state.blok is not None:
        st.write("Telah dilakukan proses split. Hasil dari split bisa dilihat sebagai berikut:")
        col1, col2, pad = st.columns([3,2,2])

        with col1:
            st.caption(f"Jumlah blok setelah split: {len(st.session_state.blok)}")
            plt.imshow(st.session_state.image_rgb, cmap='gray')
            ax = plt.gca()
            for (x, y, w, h, depth) in st.session_state.blok:
                rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=1)
                ax.add_patch(rect)
            plt.title("Hasil Split")
            st.pyplot(plt)

        with col2:
            st.write(" ")
            p_merge = st.number_input("Threshold standar deviasi (p):", min_value=0.1, max_value=0.9, value=0.7, step=0.1)
            sigma_merge = np.percentile(st.session_state.sd_map[st.session_state.sd_map >= 0], p_merge * 100)
            sd_threshold = sigma_merge ** 2
            color_threshold = st.number_input("Threshold warna:", min_value=1, max_value=100, value=40, step=1)
            h_threshold = st.number_input("Threshold homogeneity:", min_value=0.1, max_value=0.9, value=0.6, step=0.1)
            c_threshold = st.number_input("Threshold contrast:", min_value=1, max_value=10, value=5, step=1)
            e_threshold = st.number_input("Threshold energy:", min_value=0.1, max_value=0.9, value=0.6, step=0.1)
            
            if st.button("Lakukan merge"):
                if st.session_state.progress >= 36 and st.session_state.progress < 50:
                    st.session_state.progress += 12
                    progress_bar.progress(st.session_state.progress)

                class_map = merge_blocks(st.session_state.blok, 
                                         st.session_state.sd_map, 
                                         st.session_state.gray, 
                                         st.session_state.lab, 
                                         sd_threshold, 
                                         color_threshold,
                                         h_threshold,
                                         c_threshold,
                                         e_threshold)
                st.session_state.class_map = class_map
                if st.session_state.class_map is not None:
                    st.write("gambar sudah dilakukan merge! silahkan periksa di tab selanjutnya.")

    else:
            st.session_state.blok = None

with tab5:
    if st.session_state.class_map is not None:
        st.write("Telah dilakukan proses merge. Hasil dari merge bisa dilihat sebagai berikut:")
        col1, col2, pad = st.columns([3,2,2])

        with col1:
            unique_classes = np.unique(st.session_state.class_map)
            st.caption(f"Jumlah kelas berbeda: {len(unique_classes)}")
            plt.figure(figsize=(9, 5))
            plt.imshow(st.session_state.image_rgb)
            ax = plt.gca()
            unique_labels = np.unique(st.session_state.class_map)
            unique_labels = unique_labels[unique_labels != 0]

            for label in unique_labels:
                ys, xs = np.where(st.session_state.class_map == label)
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()
                rect = patches.Rectangle((x_min, y_min), x_max - x_min + 1, y_max - y_min + 1,
                                        linewidth=1.5, edgecolor='cyan', facecolor='none')
                ax.add_patch(rect)
            plt.title("Hasil Merge")
            plt.axis('off')
            st.pyplot(plt)

        with col2:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            if st.button("Lakukan eliminasi"):
                if st.session_state.progress >=48 and st.session_state.progress < 62:
                    st.session_state.progress += 12
                    progress_bar.progress(st.session_state.progress)
                
                cleaned_map = remove_small_classes(st.session_state.class_map, min_size=64)
                st.session_state.cleaned_map = cleaned_map
                if st.session_state.cleaned_map is not None:
                    st.write("gambar sudah dilakukan eliminasi! silahkan periksa di tab selanjutnya.")
   
    else:
            st.session_state.class_map = None

with tab6:
    if st.session_state.cleaned_map is not None:
        st.write("Telah dilakukan proses eliminasi. Hasil dari eliminasi bisa dilihat sebagai berikut:")
        col1, col2, pad = st.columns([3,2,2])

        with col1:
            unique_labels = np.unique(st.session_state.cleaned_map)
            num_classes = len(unique_labels[unique_labels != 0])
            st.caption(f"Jumlah blok/kelas setelah eliminasi: {num_classes}")
            plt.figure(figsize=(9, 5))
            plt.imshow(st.session_state.image_rgb)
            ax = plt.gca()

            background_mask = (st.session_state.cleaned_map == 0)
            red_overlay = np.zeros_like(st.session_state.image_rgb, dtype=np.uint8)
            red_overlay[..., 0] = 255
            ax.imshow(np.where(background_mask[..., None], red_overlay, 0), alpha=0.3)

            unique_labels = np.unique(st.session_state.cleaned_map)
            unique_labels = unique_labels[unique_labels != 0]

            for label in unique_labels:
                ys, xs = np.where(st.session_state.cleaned_map == label)
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()
                rect = patches.Rectangle((x_min, y_min), x_max - x_min + 1, y_max - y_min + 1,
                                        linewidth=1.5, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)

            plt.title("Hasil Eliminasi")
            plt.axis('off')
            st.pyplot(plt)

        with col2:
            if st.button("Lakukan region growth"):
                if st.session_state.progress >= 60 and st.session_state.progress < 75:
                    st.session_state.progress += 12
                    progress_bar.progress(st.session_state.progress)

                filled_map = combine(st.session_state.cleaned_map, st.session_state.lab)
                st.session_state.filled_map = filled_map
                if st.session_state.filled_map is not None:
                    st.write("gambar sudah dilakukan region growth! silahkan periksa di tab selanjutnya.")

    else:
            st.session_state.cleaned_map = None

with tab7:
    if st.session_state.filled_map is not None:
        st.write("Telah dilakukan proses region growth. Hasil dari region growth bisa dilihat sebagai berikut:")
        col1, col2, pad = st.columns([3,2,2])

        with col1:
            unique_labels = np.unique(st.session_state.filled_map)
            num_classes = len(unique_labels[unique_labels != 0])
            st.caption(f"Jumlah kelas: {num_classes}")
            labels = np.unique(st.session_state.filled_map)
            labels = labels[labels != 0]

            cmap = plt.get_cmap('tab20')
            colors = [cmap(i % 20) for i in range(len(labels))]

            img_viz = np.zeros(st.session_state.filled_map.shape + (3,), dtype=np.float32)

            plt.figure(figsize=(9, 5))
            ax = plt.gca()

            for i, label in enumerate(labels):
                mask = (st.session_state.filled_map == label)
                img_viz[mask] = colors[i][:3]

                ys, xs = np.where(mask)
                y_center = int(np.mean(ys))
                x_center = int(np.mean(xs))

                ax.text(x_center, y_center, str(label), color='white', fontsize=12, fontweight='bold',
                        ha='center', va='center', path_effects=[
                            plt.matplotlib.patheffects.Stroke(linewidth=2, foreground='black'),
                            plt.matplotlib.patheffects.Normal()
                        ])

            ax.imshow(img_viz)
            ax.axis('off')
            plt.title('Visualisasi Blok dengan Nomor Kelas')
            st.pyplot(plt)

        with col2:
            color_threshold_biner = st.number_input("Threshold warna:", min_value=1, max_value=100, value=25, step=1)
            if st.button("Lakukan Reclassification"):
                if st.session_state.progress >= 72 and st.session_state.progress < 87:
                    st.session_state.progress += 12
                    progress_bar.progress(st.session_state.progress)

                binary_map = simplify_to_two_classes(st.session_state.filled_map, st.session_state.lab, color_threshold_biner)
                st.session_state.binary_map = binary_map
                if st.session_state.binary_map is not None:
                    st.write("gambar sudah dilakukan Reclassification! silahkan periksa di tab selanjutnya.")

    else:
            st.session_state.filled_map = None

with tab8:
    if st.session_state.binary_map is not None:
        st.write("Telah dilakukan proses Reclassification. Hasil dari Reclassification bisa dilihat sebagai berikut:")
        col1, col2, pad = st.columns([3,2,2])

        with col1:
            labels = np.unique(st.session_state.binary_map)
            labels = labels[labels != 0]

            cmap = plt.get_cmap('tab20')
            colors = [cmap(i % 20) for i in range(len(labels))]

            img_viz = np.zeros(st.session_state.binary_map.shape + (3,), dtype=np.float32)

            plt.figure(figsize=(9, 5))
            ax = plt.gca()

            for i, label in enumerate(labels):
                mask = (st.session_state.binary_map == label)
                img_viz[mask] = colors[i][:3]

                ys, xs = np.where(mask)
                y_center = int(np.mean(ys))
                x_center = int(np.mean(xs))

                ax.text(x_center, y_center, str(label), color='white', fontsize=12, fontweight='bold',
                        ha='center', va='center', path_effects=[
                            plt.matplotlib.patheffects.Stroke(linewidth=2, foreground='black'),
                            plt.matplotlib.patheffects.Normal()
                        ])

            ax.imshow(img_viz)
            ax.axis('off')
            plt.title('Visualisasi Blok dengan Nomor Kelas')
            st.pyplot(plt)

        with col2:
            color_threshold_biner = st.number_input("Threshold warna 2:", min_value=1, max_value=100, value=25, step=1)
            if st.button("Buat visualisasi"):
                if st.session_state.progress >= 84 and st.session_state.progress < 100:
                    st.session_state.progress += 16
                    progress_bar.progress(st.session_state.progress)

                overlayed = overlay_mask(st.session_state.image_rgb, st.session_state.binary_map, alpha=0.2)
                st.session_state.overlayed = overlayed
                if st.session_state.overlayed is not None:
                    st.write("Semua tahapan sudah selesai! silahkan lihat visualisasinya di tab selanjutnya.")

    else:
            st.session_state.binary_map = None

with tab9:
    if st.session_state.overlayed is not None:
        st.write("Hasil segmentasi sawah vs non-sawah pada citra satelit:")
        col1, col2, pad = st.columns([3,3,2])

        with col1:
            plt.figure(figsize=(12, 8))

            plt.subplot(1, 1, 1)
            plt.imshow(st.session_state.image_rgb)
            plt.title('Citra Asli')
            plt.axis('off')
            st.pyplot(plt)

        with col2:
            plt.figure(figsize=(12, 8))

            plt.subplot(1, 1, 1)
            plt.imshow(st.session_state.overlayed[..., ::-1])
            plt.title('Segmentasi sawah vs non-sawah')
            plt.axis('off')
            st.pyplot(plt)

    else:
            st.session_state.overlayed = None