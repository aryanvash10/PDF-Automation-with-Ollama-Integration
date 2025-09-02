import fitz
import os

def generate_large_pdf(filename, pages=400):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        text = f"Page {i+1}: Sample energy report data. Voltage: {i*10}V, Current: {i*0.5}A, Efficiency: {0.9-i*0.001:.2f}."
        page.insert_text((50, 50), text)
    doc.save(filename)
    print(f"Generated {filename} with {pages} pages")

generate_large_pdf("E:/PDFmuTutorial/large1.pdf", 400)
generate_large_pdf("E:/PDFmuTutorial/large2.pdf", 400)