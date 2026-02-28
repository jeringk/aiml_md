import fitz
import glob
import os

pdf_files = sorted(glob.glob("Lectures/*.pdf"))

with open("/tmp/topics_extracted.md", "w", encoding="utf-8") as out:
    for pdf in pdf_files:
        try:
            doc = fitz.open(pdf)
            out.write(f"# {os.path.basename(pdf)}\n\n")
            
            topics = set()
            for page_num in range(doc.page_count):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                max_size = 0
                title = ""
                
                for b in blocks:
                    if "lines" not in b: continue
                    for l in b["lines"]:
                        for s in l["spans"]:
                            size = s["size"]
                            text = s["text"].strip()
                            if text and "BITS" not in text and "Garima" not in text and "Lecture" not in text and "Questions" not in text and "Recap" not in text and "Social Media" not in text and "Agenda" not in text:
                                if size > max_size and len(text) > 3:
                                    max_size = size
                                    title = text
                
                if title:
                    if title not in topics and len(title) < 100:
                        topics.add(title)
                        out.write(f"- {title}\n")
                        
            out.write("\n")
        except Exception as e:
            out.write(f"Error reading {pdf}: {e}\n\n")
