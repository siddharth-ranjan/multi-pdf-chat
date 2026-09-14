import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_pdf(pages):
    """Builds a minimal PDF with real text; each page is a list of lines (empty list = blank page)."""
    objects = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids, number = [], 4
    for lines in pages:
        stream = ("BT /F1 12 Tf 72 720 Td 16 TL " + " ".join(f"({line}) Tj T*" for line in lines) + " ET") if lines else ""
        objects[number] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
        objects[number + 1] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {number} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>"
        )
        kids.append(f"{number + 1} 0 R")
        number += 2
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out = b"%PDF-1.4\n"
    offsets = {}
    for key in sorted(objects):
        offsets[key] = len(out)
        out += f"{key} 0 obj\n{objects[key]}\nendobj\n".encode()
    xref = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offsets[key]:010d} 00000 n \n" for key in range(1, size)).encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out
