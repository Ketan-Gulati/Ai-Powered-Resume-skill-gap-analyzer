import pdfplumber, io

def extract_pdf_text(file_bytes):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join([p.extract_text() or "" for p in pdf.pages])