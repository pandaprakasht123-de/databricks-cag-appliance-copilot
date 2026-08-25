# Databricks notebook source
# MAGIC %pip install pdfplumber
# MAGIC %restart_python

# COMMAND ----------

import os
import re
import pdfplumber
from pyspark.sql.functions import current_timestamp

# Define paths
pdf_volume_path = "/Volumes/llm/rag/pdf_volume/"

# 1. Read already processed files to avoid duplicate work
processed_files_df = spark.sql("SELECT DISTINCT file_name FROM llm.rag.docs_track_multiple_model")
processed_files = set(row["file_name"] for row in processed_files_df.collect())

# 2. Identify new files
all_volume_files = [f for f in os.listdir(pdf_volume_path) if f.endswith(".pdf")]
new_files = [f for f in all_volume_files if f not in processed_files]

print(f"New files detected to ingest: {new_files}")

# 3. Parse PDFs and extract full text per model/version
extracted_data = []

for file_name in new_files:
    file_path = os.path.join(pdf_volume_path, file_name)
    
    # Extract model and version from filename (e.g. WM100_v1.pdf -> model: WM100, version: v1)
    match = re.match(r"([A-Za-z0-9]+)_([vV0-9\.]+)\.pdf", file_name)
    if match:
        model_number = match.group(1)
        manual_version = match.group(2)
    else:
        model_number = "UNKNOWN"
        manual_version = "UNKNOWN"

    full_pdf_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            full_pdf_text += page_text + "\n"

    extracted_data.append({
        "file_name": file_name,
        "model_number": model_number,
        "manual_version": manual_version,
        "text": full_pdf_text.strip()
    })

if extracted_data:
    # 4. Convert to Spark DataFrame
    new_docs_df = spark.createDataFrame(extracted_data)
    
    # 5. Insert into docs_text_multiple_model table
    new_docs_df.select("file_name", "model_number", "manual_version", "text") \
        .write.mode("append").saveAsTable("llm.rag.docs_text_multiple_model")
    
    # 6. Track processed files in docs_track_multiple_model table
    track_df = new_docs_df.select("file_name", "model_number", "manual_version") \
        .withColumn("load_timestamp", current_timestamp())
    
    track_df.write.mode("append").saveAsTable("llm.rag.docs_track_multiple_model")
    print(f"Successfully processed and cataloged {len(extracted_data)} manuals.")
else:
    print("No new documents to process.")