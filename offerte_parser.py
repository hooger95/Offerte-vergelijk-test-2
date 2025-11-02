import streamlit as st
from pathlib import Path
from offerte_vergelijker import compare_pdf_offers

st.title("Offerte Vergelijker")

st.markdown("Upload twee offertes in PDF-formaat om ze te vergelijken.")

pdf_old = st.file_uploader("Oude offerte (PDF)", type=["pdf"])
pdf_new = st.file_uploader("Nieuwe offerte (PDF)", type=["pdf"])

price_tol = st.slider("Prijsverschil tolerantie (€)", min_value=0.00, max_value=5.00, value=0.02, step=0.01)

if pdf_old and pdf_new:
    if st.button("Vergelijk offertes"):
        tmp_dir = Path("tmp_uploads")
        tmp_dir.mkdir(exist_ok=True)
        old_path = tmp_dir / pdf_old.name
        new_path = tmp_dir / pdf_new.name
        with open(old_path, "wb") as f:
            f.write(pdf_old.read())
        with open(new_path, "wb") as f:
            f.write(pdf_new.read())

        xlsx_path, txt_path = compare_pdf_offers(str(old_path), str(new_path), out_prefix="streamlit_vergelijk", price_tol=price_tol)

        st.success("Vergelijking voltooid!")
        st.download_button("Download Excel-rapport", data=open(xlsx_path, "rb").read(), file_name="vergelijking.xlsx")
        st.download_button("Download tekstsamenvatting", data=open(txt_path, "rb").read(), file_name="vergelijking.txt")
