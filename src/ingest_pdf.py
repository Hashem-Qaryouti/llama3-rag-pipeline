from PyPDF2 import PdfReader
import paths

from paths import PDF_PATH

def load_pdf(pdf_path=PDF_PATH):
    """ This function reads the PDF and returns all text as a single string

        Params:
            pdf_path (str): represents the pdf file path

        Returns:
            returns all text as a single string
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        # Extract text from each page and append it to text var
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
        return text
    
if __name__ == "__main__":
    pdf_text = load_pdf()
    print("PDF loaded. First 500 characters:")
    print(pdf_text[:500])