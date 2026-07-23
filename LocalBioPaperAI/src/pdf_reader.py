import fitz


def read_pdf(pdf_path):
    """
    Liest eine PDF-Datei und gibt den gesamten Text zurück.
    """

    document = fitz.open(pdf_path)

    full_text = ""

    for page in document:
        full_text += page.get_text()

    document.close()

    return full_text