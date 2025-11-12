import pdfplumber

def extract_text_from_pdf(fobj):
    try:
        pages = []
        with pdfplumber.open(fobj) as pdf:
            for p in pdf.pages:
                txt = p.extract_text() or ""
                pages.append(txt)
        return "\n".join(pages).strip()
    except Exception:
        try:
            data = fobj.read().decode('utf8', errors='replace')
            return data[:10000]
        except Exception:
            return ""