from PyPDF2 import PdfReader
import paths
import os
print("Current working directory:", os.getcwd())
from paths import PDF_PATH


def load_pdf():
    reader = PdfReader(PDF_PATH)
    print(f"Number of pages in PDF: {len(reader.pages)}")  # check pages
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
        else:
            print(f"Warning: Page {i+1} has no text extracted.")
    print(f"Total characters extracted: {len(text)}")
    return text

if __name__ == "__main__":
    pdf_text = load_pdf()
