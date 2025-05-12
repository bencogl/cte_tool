import os
import re
import uuid
import shutil
import json

import gradio as gr
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import pandas as pd
from fuzzywuzzy import process
from transformers import pipeline

# 1) Definisci i pattern dei campi CTE
CTE_PATTERNS = [
    r'Prezzo Luce',
    r'Prezzo.*perdite',
    r'Prezzo gas',
    r'Fee.*Listino.*PrimoAnno',
    r'PCV Fissa Listino',
    r'QVD Fissa Listino',
    r'Fee Gas Listino',
    r'Fee Primo Anno',
    r'Sconto Fedeltà',
    r'Perdite di rete',
    r'Indice GO',
    r'PrezzoMisuratore'
]

# 2) Inizializza il modello Hugging Face (FLAN-T5 small, gratuito)
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    tokenizer="google/flan-t5-small"
)

def process_files(pdfs, excel):
    # --- 3.1) Crea cartella temporanea univoca per questo run
    run_id = str(uuid.uuid4())
    os.makedirs(run_id, exist_ok=True)

    # --- 3.2) Salva i PDF e l’Excel sul disco
    pdf_paths = []
    for pdf in pdfs:
        path = os.path.join(run_id, pdf.name)
        with open(path, "wb") as f:
            f.write(pdf.read())
        pdf_paths.append(path)

    excel_path = os.path.join(run_id, excel.name)
    with open(excel_path, "wb") as f:
        f.write(excel.read())

    # --- 3.3) Carica e verifica l’Excel
    df = pd.read_excel(excel_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("codice item", "codice listino", "prezzo unitario"):
        if col not in df.columns:
            # pulizia e messaggio di errore
            shutil.rmtree(run_id)
            return f"❌ Errore: manca colonna '{col}' nell'Excel.", None

    all_reports = []
    # --- 3.4) Per ciascun PDF:
    for pdf_path in pdf_paths:
        # 3.4.1) Converti la 2ª pagina in immagine
        images = convert_from_path(pdf_path, first_page=2, last_page=2, dpi=300)
        img_path = pdf_path.replace(".pdf", ".png")
        images[0].save(img_path, "PNG")

        # 3.4.2) OCR sull’immagine
        text = pytesseract.image_to_string(Image.open(img_path))
        lines = [l for l in text.splitlines() if l.strip()]

        # 3.4.3) Estrai Codice Listino e Codice Prodotto
        codice_listino = next(
            (l.strip() for l in lines if re.match(r'^[A-Z0-9_]{5,}$', l.strip())),
            None
        )
        codice_prodotto = lines[0].strip() if lines else None

        # 3.4.4) Estrai tutti i campi CTE
        valori = {}
        for l in lines:
            for pat in CTE_PATTERNS:
                if re.search(pat, l, re.IGNORECASE):
                    parts = re.split(r'\s{2,}|: ', l, maxsplit=1)
                    if len(parts) == 2:
                        valori[parts[0].strip()] = parts[1].strip()
                    break

        # 3.4.5) Confronta con l’Excel
        sub = df[df["codice listino"] == codice_listino]
        expected = {
            row["codice item"]: row["prezzo unitario"]
            for _, row in sub.iterrows()
        }

        # Costruisci la tabella Markdown di confronto
        table_md = "|Campo PDF|PDF estratto|Excel atteso|Stato|\n"
        table_md += "|---|---|---|---|\n"
        for campo, vpdf in valori.items():
            match, score = process.extractOne(
                campo, list(expected.keys()) or [""], score_cutoff=80
            )
            if not match:
                table_md += f"|{campo}|{vpdf or ''}|—|Non trovato|\n"
            else:
                vxl = expected[match]
                stato = "OK" if vpdf == vxl else "Diverso"
                table_md += f"|{campo}|{vpdf}|{vxl}|{stato}|\n"

        # Raccogli il report
        all_reports.append({
            "nome_file": os.path.basename(pdf_path),
            "codice_listino": codice_listino,
            "codice_prodotto": codice_prodotto,
            "valori_estratti": valori,
            "tabella_confronto_md": table_md
        })

    # --- 3.5) Prepara il prompt per l’LLM
    prompt = (
        "Genera un report utente-friendly in Markdown "
        "basato sui seguenti report JSON raw:\n\n"
        + json.dumps(all_reports, ensure_ascii=False, indent=2)
    )

    # 3.5) Chiama il modello HF per formattare il report finale
    out = generator(
        prompt,
        max_length=1024,
        do_sample=False
    )[0]["generated_text"]

    # --- 3.6) Pulisci la cartella temporanea
    shutil.rmtree(run_id)

    return out, all_reports

# --- 3.7) Costruzione interfaccia Gradio
with gr.Blocks() as demo:
    gr.Markdown("# ListinoChecker (luce & gas)\nCarica PDF + Excel e ottieni il report.")
    pdfs = gr.File(label="PDF di offerte", file_count="multiple", file_types=[".pdf"])
    excel = gr.File(label="File Excel Tracciato_CED", file_count=1, file_types=[".xlsx", ".xls"])
    btn = gr.Button("Processa")
    output_md = gr.Markdown()
    output_json = gr.JSON(label="Reports raw")
    btn.click(
        fn=process_files,
        inputs=[pdfs, excel],
        outputs=[output_md, output_json]
    )
demo.launch(share=True)
