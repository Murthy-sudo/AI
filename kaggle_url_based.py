import streamlit as st
import requests
import os
import zipfile

st.title("📦 Kaggle Dataset Downloader")

# --- Config ---
file_path = r"C:\Users\GenAIHYDSYPUSR117\Desktop\BTTEAM\dataset.zip"
extract_path = r"C:\Users\GenAIHYDSYPUSR117\Desktop\BTTEAM\data"
headers = {"Authorization": "Bearer KGAT_ce8c9aa9a0a38f13958d5a7afa243fe1"}  # <-- Replace with your Kaggle API token

# --- Search field ---
search_key = st.text_input("🔍 Enter keyword to search Kaggle datasets:")

if search_key:
    try:
        url = "https://www.kaggle.com/api/v1/datasets/list"
        params = {
            "page": 1,
            "pageSize": 5,   # only 5 datasets
            "search": search_key,
            "fileType": "csv",
            "sortBy": "hottest"
        }

        response = requests.get(url, headers=headers, verify=False, params=params)
        response.raise_for_status()
        data = response.json()

        st.subheader("Top 5 datasets found:")
        for ds in data:   # ✅ loop directly over the list
            st.write(f"**Title:** {ds.get('title')}")
            st.write(f"**Ref:** {ds.get('ref')}")
            st.write(f"**URL:** {ds.get('url')}")
            st.write(f"**Creator:** {ds.get('creatorName')}")
            st.write(f"**Size:** {ds.get('totalBytes')/1024:.2f} KB")
            st.write(f"**Last Updated:** {ds.get('lastUpdated')}")
            st.write("---")
    except Exception as e:
        st.error(f"❌ Error while searching: {e}")

# --- Dataset URL input ---
dataset_url = st.text_input("📥 Enter Kaggle dataset download URL:")

if dataset_url:
    try:
        response = requests.get(dataset_url, headers=headers, verify=False, stream=True)
        response.raise_for_status()

        # Save the zip file
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        st.success(f"✅ Successfully downloaded dataset to: {file_path}")

        # --- UNZIP LOGIC ---
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        st.success(f"📂 Files extracted to: {extract_path}")

    except Exception as e:
        st.error(f"❌ Error while downloading: {e}")
