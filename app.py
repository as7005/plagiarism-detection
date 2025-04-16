# src/app.py

import streamlit as st
from utils import detect_plagiarism

st.set_page_config(page_title="Plagiarism Checker", layout="centered")
st.title("🧠 Plagiarism Detection System")

text1 = st.text_area("Enter the first text", height=200)
text2 = st.text_area("Enter the second text", height=200)

if st.button("Check Plagiarism"):
    if text1.strip() == "" or text2.strip() == "":
        st.warning("Please enter both texts to compare.")
    else:
        with st.spinner("Analyzing..."):
            results = detect_plagiarism(text1, text2)
        st.success("Analysis Complete!")

        st.subheader("📊 Similarity Scores")
        st.write(f"**TF-IDF Cosine Similarity:** {results['TF-IDF Cosine Similarity']:.4f}")
        st.write(f"**N-gram Overlap (Trigram):** {results['N-gram Overlap (trigram)']:.4f}")
        st.write(f"**Jaccard Similarity:** {results['Jaccard Similarity']:.4f}")
        st.write(f"**USE Cosine Similarity:** {results['USE Cosine Similarity']:.4f}")

        st.subheader("🔍 Semantic Role Labels")
        st.markdown("**Text 1 Roles:**")
        st.write(results["Semantic Roles Text1"])
        st.markdown("**Text 2 Roles:**")
        st.write(results["Semantic Roles Text2"])
