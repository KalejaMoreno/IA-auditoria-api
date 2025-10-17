import mimetypes
from pathlib import Path

def detect_file_type(file_path):
    ext = Path(file_path).suffix.lower()

    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.svg']:
        return 'image'
    elif ext in ['.pdf']:
        return 'pdf'
    elif ext in ['.doc', '.docx']:
        return 'word'
    elif ext in ['.ppt', '.pptx']:
        return 'powerpoint'
    elif ext in ['.zip']:
        return 'zip'
    else:
        return 'unsupported'
