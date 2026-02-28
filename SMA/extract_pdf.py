import fitz
import os
import glob

pdf_files = glob.glob("Lectures/*.pdf")
for pdf in pdf_files:
    try:
        doc = fitz.open(pdf)
        print(f"--- {pdf} ---")
        # Extract text from first 3 pages to get topics
        text = ""
        for i in range(min(3, doc.page_count)):
            text += doc[i].get_text()
        # Get just non-empty lines, unique ones to summarize topics
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:15]:
            print(line)
        print("======\n")
    except Exception as e:
        print(f"Failed {pdf}: {e}")
