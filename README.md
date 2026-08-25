# Databricks CAG Appliance Copilot

This project implements a Cache-Augmented Generation (CAG) architecture for appliance technical support using Databricks, Delta tables, LangChain, and MLflow model registration in Unity Catalog.

Unlike a traditional vector-search RAG pattern, this solution keeps the full appliance manual as the retrieval unit and selects the correct document by model and version before sending it to the model. The approach avoids embeddings, vector indexes, and chunking loss while preserving document structure and version accuracy.

## Architecture overview

The architecture in [Docs/Architecture.png](Docs/Architecture.png) follows a three-layer model:

1. Document Management Layer
   - Stores appliance manuals in a Databricks volume
   - Tracks file names and manual versions
   - Preserves full PDFs as source-of-truth documents

2. CAG Context Assembly Layer
   - Detects model and version information from filenames
   - Filters by metadata such as model number and manual version
   - Retrieves the full document instead of chunked embeddings
   - Eliminates version drift by always selecting the correct manual version

3. LLM Inference Layer
   - Builds a prompt using the selected manual + user question + system instructions
   - Calls a Databricks-hosted foundation model
   - Produces a grounded answer with the correct manual context

## Project purpose

The main goal is to help users get support insights from appliance manuals such as:
- model-specific troubleshooting guidance
- version-specific differences in manuals
- accurate answers grounded in the correct appliance document
- lower operational complexity than vector-index systems

The project is designed for a Databricks environment and uses Unity Catalog as the governance and storage layer for tables, volumes, functions, and registered models.

## Unity Catalog assets used

This project is built around the following Databricks Unity Catalog resources:

- Catalog: `main` / `llm` (depending on the Databricks workspace configuration)
- Schema: `rag`
- Tables:
  - `llm.rag.docs_text_multiple_model`
  - `llm.rag.docs_track_multiple_model`
- Volume:
  - `/Volumes/llm/rag/pdf_volume/`
- Functions:
  - `llm.rag.cag_appliance_chatbot`
  - `llm.rag.my_demo_chatbot`
- Registered model:
  - `llm.rag.cag_appliance_chatbot`

These objects allow the solution to keep source data, processed text, ingestion tracking, and the final deployed model under a managed and governed Unity Catalog structure.

## Unity Catalog workflow

1. Raw PDF manuals are stored in a Unity Catalog volume.
2. The notebook reads and processes those files into Delta tables in the `llm.rag` schema.
3. Tracking metadata is stored in `docs_track_multiple_model` to prevent duplicate ingestion.
4. A custom Python model is logged and registered in Unity Catalog using `mlflow.set_registry_uri("databricks-uc")`.
5. The final model can be reused or deployed from the Unity Catalog model registry.

## Repository structure

```text
.
├── Data/
│   ├── WM100_v1.pdf
│   ├── WM100_v2.pdf
│   ├── WM200_v1.pdf
│   ├── WM300_v1.pdf
│   ├── WM400_v1.pdf
│   ├── WM400_v2.pdf
│   ├── WM500_v1.pdf
│
├── Docs/
│   └── Architecture.png
│
├── Notebooks/
│   ├── 1.create_tables.py
│   ├── 2.Incremental_pdf_to_dcos_text.py
│   └── 3.CAG_Chatbot_Engine_&_MLflow_UC_Logging.py
│
├── README.md
├── requirement.txt
└── .gitignore
```

## Notebook workflow

### 1) Create tables
File: [Notebooks/1.create_tables.py](Notebooks/1.create_tables.py)

This notebook creates the Delta tables used to store:
- extracted document text by model and manual version
- tracking metadata for already-processed files

Tables created:
- `llm.rag.docs_text_multiple_model`
- `llm.rag.docs_track_multiple_model`

### 2) Ingest PDFs and extract text
File: [Notebooks/2.Incremental_pdf_to_dcos_text.py](Notebooks/2.Incremental_pdf_to_dcos_text.py)

This notebook:
- reads PDFs from `/Volumes/llm/rag/pdf_volume/`
- extracts text using `pdfplumber`
- identifies model name and manual version from file naming convention like `WM100_v1.pdf`
- appends extracted text to the main document table
- avoids reprocessing files already tracked in the ingestion table

This is an incremental pipeline that only processes newly added manuals.

### 3) Build the support chatbot and register MLflow model
File: [Notebooks/3.CAG_Chatbot_Engine_&_MLflow_UC_Logging.py](Notebooks/3.CAG_Chatbot_Engine_&_MLflow_UC_Logging.py)

This notebook:
- fetches document text from Delta tables
- builds a knowledge corpus for the full appliance manual set
- creates a system prompt that instructs the model to answer using only the provided manuals
- calls `ChatDatabricks` with a model endpoint such as `databricks-meta-llama-3-3-70b-instruct`
- tests example user queries
- wraps the chatbot in a custom `mlflow.pyfunc.PythonModel`
- logs and registers the model to Unity Catalog as `llm.rag.cag_appliance_chatbot`

## Data flow

1. PDF manuals are placed in a Databricks volume and organized by model/version naming convention such as `WM100_v2.pdf`.
2. The ingestion notebook reads the files, extracts full document text, and stores the result in Delta tables.
3. The chatbot notebook loads the full document set and filters based on user-selected model and manual version.
4. The selected manual is passed to a prompt builder together with the user question and system instructions.
5. The foundation model answers using the exact correct manual context rather than a vector search index or embeddings.
6. The final chatbot is packaged and registered in MLflow/Unity Catalog for deployment or reuse.

## Why this design is different

This architecture intentionally follows the CAG pattern visible in [Docs/Architecture.png](Docs/Architecture.png):
- no vector database
- no embedding pipeline
- no chunking or chunk-loss risk
- full-document retrieval by model and version
- lower maintenance and easier governance

This is especially useful for appliance support scenarios where the same error code can have different meanings across model revisions.

## Dependencies

The project depends on:
- `databricks-langchain`
- `langchain`
- `langchain-core`
- `mlflow`
- `databricks-sdk`
- `pdfplumber`
- `pandas`

See [requirement.txt](requirement.txt) for the project dependency list.

## Prerequisites

Before running this project, ensure the following are available in your Databricks workspace:
- a Databricks cluster with Python support
- Unity Catalog enabled
- access to the target catalog/schema such as `llm.rag`
- a mounted or accessible volume containing PDF manuals
- a valid Databricks model serving endpoint or supported LLM endpoint

## Databricks job trigger for new file arrival

This workflow can be automated with a Databricks job that runs whenever a new PDF file is uploaded to the catalog volume.

### Recommended trigger pattern

1. Place new appliance manuals into `/Volumes/llm/rag/pdf_volume/`.
2. Configure a Databricks job to monitor that volume or run on a scheduled interval.
3. Trigger the ingestion notebook, [Notebooks/2.Incremental_pdf_to_dcos_text.py](Notebooks/2.Incremental_pdf_to_dcos_text.py).
4. The notebook checks which files have not yet been processed using the tracking table.
5. Only new files are extracted, inserted into `llm.rag.docs_text_multiple_model`, and logged in `llm.rag.docs_track_multiple_model`.
6. The chatbot logic can then use the updated knowledge base without reprocessing existing manuals.

### Job design idea

- Trigger type: file arrival event or scheduled job every 5–15 minutes
- Notebook: [Notebooks/2.Incremental_pdf_to_dcos_text.py](Notebooks/2.Incremental_pdf_to_dcos_text.py)
- Output: updated Delta tables and fresh knowledge base for the chatbot
- Safety: use the tracking table to avoid duplicate ingestion and maintain idempotent behavior

This setup makes the solution automatically refresh when new manuals arrive, which is essential for keeping the appliance support knowledge current.

## Example usage

The chatbot is intended to answer questions like:
- What does error E21 mean on WM100 version v2?
- What does SUDS mean in model WM400 v2?

The prompt logic explicitly requires the user to identify the model number and version before giving troubleshooting guidance.

## Notes

- This project is focused on appliance manuals and support guidance.
- Manual versioning is essential because different revisions may contain different troubleshooting instructions.
- The system is designed to reject requests outside the provided documentation set.

## Future enhancements

Possible extensions include:
- chunked document retrieval using vector search
- semantic caching for repeated queries
- logging and monitoring of user questions and answers
- deployment as a REST API or serving endpoint
- support for additional appliance brands or product families
