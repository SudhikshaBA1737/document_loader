import os
import urllib.request
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sample PDF URLs
PDF_FILES = {
    "sample1.pdf": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "sample3.pdf": "https://www.irs.gov/pub/irs-pdf/fw9.pdf"
}

def download_sample_pdfs():
    for filename, url in PDF_FILES.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")

def load_pdfs(pdf_paths):
    documents = []
    for path in pdf_paths:
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        else:
            print(f"Warning: {path} not found")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# ------------------------------------------
# Exercise 3 — Attach Metadata
# ------------------------------------------
def attach_metadata(chunks, source_mapping):
    upload_date = datetime.now().strftime("%Y-%m-%d")
    enriched_chunks = []

    for chunk in chunks:
        # Copy chunk to avoid modifying original
        try:
            new_chunk = chunk.model_copy(deep=True)
        except AttributeError:
            new_chunk = chunk.copy(deep=True)

        original_source = chunk.metadata.get("source", "unknown.pdf")
        filename = os.path.basename(original_source)
        page_number = chunk.metadata.get("page", 0) + 1

        source_type = source_mapping.get(filename, "unknown")

        new_chunk.metadata.update({
            "filename": filename,
            "page_number": page_number,
            "upload_date": upload_date,
            "source_type": source_type
        })

        enriched_chunks.append(new_chunk)

    return enriched_chunks

# ------------------------------------------
# Exercise 4 — Filter Function (Improved)
# ------------------------------------------
def filter_chunks(chunks, **filters):
    matched_chunks = []

    for chunk in chunks:
        match = True
        for key, value in filters.items():
            # Case-insensitive comparison
            if str(chunk.metadata.get(key, "")).lower() != str(value).lower():
                match = False
                break
        if match:
            matched_chunks.append(chunk)

    return matched_chunks

# ------------------------------------------
# Exercise 5 — Test Your Filters
# ------------------------------------------
def main():
    # Step 0: Download PDFs
    download_sample_pdfs()

    # Step 1: Load PDFs
    print("\n--- Loading PDFs ---")
    pdf_files = list(PDF_FILES.keys())
    documents = load_pdfs(pdf_files)
    print(f"Loaded {len(documents)} pages")

    if not documents:
        print("No documents loaded")
        return

    # Step 2: Split into chunks
    print("\n--- Splitting into Chunks ---")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Step 3: Attach metadata
    print("\n--- Attaching Metadata ---")
    source_mapping = {
        "sample1.pdf": "notes",
        "sample3.pdf": "tax form"
    }

    enriched_chunks = attach_metadata(chunks, source_mapping)

    print("\nSample metadata:")
    print(enriched_chunks[0].metadata)

    # Step 4 & 5: Filtering tests
    print("\n--- Testing Filters ---")

    # Test A
    tax_form_chunks = filter_chunks(enriched_chunks, source_type="tax form")
    print(f"\nTax form chunks: {len(tax_form_chunks)}")

    if tax_form_chunks:
        print("\nPreview:")
        # Safe print for Windows terminals to avoid cp1252 encoding errors
        snippet = tax_form_chunks[0].page_content[:200]
        print(snippet.encode("ascii", errors="replace").decode("ascii"))

    # Test B
    specific_chunks = filter_chunks(
        enriched_chunks,
        filename="sample3.pdf",
        page_number=2
    )
    print(f"\nSpecific page chunks: {len(specific_chunks)}")

    # Test C
    no_match = filter_chunks(enriched_chunks, source_type="textbook")
    print(f"\nNo match chunks: {len(no_match)}")

# ------------------------------------------
# Run
# ------------------------------------------
if __name__ == "__main__":
    main()