from pathlib import Path

import openpyxl


def parse_xlsx(path: str | Path) -> list[dict]:
    """Return one record per sheet, matching parse_pdf's page-record shape.

    Each sheet is treated as one "page" so citations can point at
    "sheet 2" the same way PDF citations point at a page number.
    """

    xlsx_path = Path(path)
    pages: list[dict] = []

    try:
        workbook = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
        for sheet_index, sheet in enumerate(workbook.worksheets):
            lines = []
            for row in sheet.iter_rows(values_only=True):
                # Skip fully empty rows so blank formatting doesn't bloat the text.
                cells = [str(cell) for cell in row if cell is not None]
                if cells:
                    lines.append(" | ".join(cells))
            text = f"Sheet: {sheet.title}\n" + "\n".join(lines)
            pages.append(
                {
                    "text": text.strip(),
                    "page_number": sheet_index + 1,
                    "tables": [],
                    "metadata": {"is_likely_scan": False, "sheet_name": sheet.title},
                }
            )
        workbook.close()
    except Exception as exc:
        raise RuntimeError(f"Failed to parse XLSX {xlsx_path}: {exc}") from exc

    return pages
