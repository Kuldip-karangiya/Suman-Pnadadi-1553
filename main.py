import streamlit as st
import os
import shutil
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow as tf
import pickle
from numpy.linalg import norm 
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import filedialog
import tqdm

# Image path for the logo
logo_path = 'Logo final.png'

col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo_path, width=200)  # Increase the width parameter to increase the size of the logo image

with col2:
    st.title('')

def extract_features(img_path, model):
    img = Image.open(img_path).convert("L")  # Open image in grayscale mode
    img = img.convert("RGB")  # Convert grayscale to RGB
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def train_model(folder_path):
    Image.MAX_IMAGE_PIXELS = 202662810
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])

    filenames = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

    feature_list = []

    for file in tqdm.tqdm(filenames):
        feature_list.append(extract_features(file, model))

    embeddings_path = os.path.join(folder_path, 'embeddings.pkl')
    filenames_path = os.path.join(folder_path, 'filenames.pkl')
    pickle.dump(feature_list, open(embeddings_path, 'wb'))
    pickle.dump(filenames, open(filenames_path, 'wb'))

    return model, np.array(feature_list), filenames

def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            while True:
                chunk = uploaded_file.read(10 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return file_path
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def feature_extraction(img_path, model):
    img = Image.open(img_path).convert("L")  # Open image in grayscale mode
    img = img.convert("RGB")  # Convert grayscale to RGB
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = min(20, len(feature_list))
    nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='brute', metric='euclidean')
    nbrs.fit(feature_list)
    distances, indices = nbrs.kneighbors([features])
    return indices

# Initialize directories
uploaded_folders_dir = 'UploadedFolders'
if not os.path.exists(uploaded_folders_dir):
    os.makedirs(uploaded_folders_dir)

# Dropdown for selecting a folder
folder_options = [d for d in os.listdir(uploaded_folders_dir) if os.path.isdir(os.path.join(uploaded_folders_dir, d))]
folder_options.insert(0, 'Upload New Folder')
selected_folder = st.selectbox('Select a Folder', folder_options)

folder_path = ""
if selected_folder == 'Upload New Folder':
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    st.write('Choose an Image folder:')
    clicked = st.button('Browse Folder')

    if clicked:
        dirname = str(filedialog.askdirectory(master=root))
        if not dirname:
            st.warning("No folder selected. Please select a folder.")
        else:
            folder_name = os.path.basename(dirname)
            destination_folder = os.path.join(uploaded_folders_dir, folder_name)
            shutil.rmtree(destination_folder, ignore_errors=True)

            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            images_reports = [file for file in os.listdir(dirname) if os.path.splitext(file)[1].lower() in ['.jpg', '.jpeg', '.png', '.gif']]
            errors = []

            for file in images_reports:
                try:
                    shutil.copy(os.path.join(dirname, file), destination_folder)
                except Exception as e:
                    errors.append(f"Error copying file '{file}': {str(e)}")

            if not errors:
                with st.spinner("Please be patient while we train the model..."):
                    model, feature_list, filenames = train_model(destination_folder)
                st.success("Folder Upload successfully ✅")
                folder_path = destination_folder
                st.experimental_rerun()  # Refresh the page to update the dropdown
            else:
                st.error("Error while copying files..! Please try again.")
else:
    folder_path = os.path.join(uploaded_folders_dir, selected_folder)

# Load pre-trained model and features if folder_path is set
if folder_path:
    embeddings_path = os.path.join(folder_path, 'embeddings.pkl')
    filenames_path = os.path.join(folder_path, 'filenames.pkl')

    feature_list = np.array(pickle.load(open(embeddings_path, 'rb')))
    filenames = pickle.load(open(filenames_path, 'rb'))

    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])

    uploaded_file = st.file_uploader('Choose an Image file')

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        if file_path is not None:
            st.image(uploaded_file)
            check_clicked = st.button("Check")
            if check_clicked:
                st.write("Generating recommendations...")
                with st.spinner("Please wait while recommendations are generated..."):
                    feature = feature_extraction(file_path, model)
                    indices = recommend(feature, feature_list)
                st.success("Recommendations generated!")

                num_rows = 4
                num_cols = 5

                for i in range(num_rows):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        index = i * num_cols + j
                        if index < len(indices[0]):
                            cols[j].image(filenames[indices[0][index]], use_column_width=True)
                            cols[j].write(os.path.basename(filenames[indices[0][index]]))
                        else:
                            cols[j].write("")  # Empty placeholder for cells without images
