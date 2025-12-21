"""
Configuração do Tesseract OCR.
"""

import os
import platform
import pytesseract


def setup_tesseract():
    """Configura o path do Tesseract (Linux e Windows)."""
    if platform.system() == 'Windows':
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"Tesseract encontrado em: {path}")
                return True
        print("AVISO: Tesseract não encontrado. Configure o caminho manualmente.")
        return False
    else:
        print("Sistema Linux/Mac detectado. Tesseract deve estar no PATH.")
        try:
            pytesseract.get_tesseract_version()
            print("Tesseract encontrado e funcionando.")
            return True
        except:
            print("AVISO: Tesseract não encontrado. Instale com: sudo apt-get install tesseract-ocr")
            return False

