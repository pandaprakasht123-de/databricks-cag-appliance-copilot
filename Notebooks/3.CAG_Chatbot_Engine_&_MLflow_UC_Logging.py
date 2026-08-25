# Databricks notebook source
# MAGIC %pip install -U databricks-langchain langchain langchain-core mlflow
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.models import infer_signature

# COMMAND ----------


# 1. Fetch documentation from Delta Table
docs_df = spark.sql("""
    SELECT model_number, manual_version, text 
    FROM llm.rag.docs_text_multiple_model
    ORDER BY model_number, manual_version
""").collect()

# 2. Build Bounded Context Corpus
knowledge_corpus = ""
for row in docs_df:
    knowledge_corpus += f"\n--- START MANUAL: MODEL {row['model_number']} (VERSION {row['manual_version']}) ---\n"
    knowledge_corpus += row['text']
    knowledge_corpus += f"\n--- END MANUAL: MODEL {row['model_number']} ---\n"

# 3. System Prompt
SYSTEM_PROMPT = f"""You are an enterprise technical support specialist for home appliances.
You have the complete technical manuals for all appliance models and versions listed below.

KNOWLEDGE BASE:
{knowledge_corpus}

INSTRUCTIONS:
1. Always identify the exact Model Number and Version specified by the user.
2. If the user does not mention their model or version, explicitly ask them to clarify before troubleshooting.
3. Distinguish between different versions of the same model (e.g., WM100 v1.0 vs WM100 v2.0).
4. If a question is outside these manuals, decline politely.
5. Provide concise, step-by-step diagnostic instructions in English.
"""

# 4. Initialize Chat Model
chat_model = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    max_tokens=350,
    temperature=0.1
)

# 5. Inference Function
def ask_cag_assistant(user_query: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query)
    ]
    response = chat_model.invoke(messages)
    return response.content

# COMMAND ----------

# Test Queries
print(ask_cag_assistant("What does error E21 mean on WM100 model vesion v2"))

# COMMAND ----------

# Test Queries
print(ask_cag_assistant("What does SUDS mean in model wm400 v2"))

# COMMAND ----------

# 1. Define the Custom PyFunc Class with the updated import
class CAGApplianceBot(mlflow.pyfunc.PythonModel):
    def __init__(self, system_prompt: str, endpoint_name: str):
        self.system_prompt = system_prompt
        self.endpoint_name = endpoint_name

    def load_context(self, context):
        # ✅ Import from databricks_langchain instead of langchain_community
        from databricks_langchain import ChatDatabricks
        self.chat_model = ChatDatabricks(
            endpoint=self.endpoint_name,
            max_tokens=350,
            temperature=0.1
        )

    def predict(self, context, model_input):
        from langchain_core.messages import HumanMessage, SystemMessage
        import pandas as pd

        # Support both Dict and Pandas DataFrame inputs from Serving Endpoints
        if isinstance(model_input, pd.DataFrame):
            query = model_input["query"].iloc[0]
        elif isinstance(model_input, dict):
            query = model_input["query"]
        else:
            query = str(model_input)

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query)
        ]
        response = self.chat_model.invoke(messages)
        return response.content

# 2. Re-log and Register to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
registered_model_name = "llm.rag.cag_appliance_chatbot"

sample_input = {"query": "What does error E21 mean on WM200 v1.0?"}
sample_output = "E21 on WM200 v1.0 indicates a water pump communication error."

with mlflow.start_run(run_name="cag_multimodel_appliance_fix") as run:
    signature = infer_signature(sample_input, sample_output)
    
    mlflow.pyfunc.log_model(
        artifact_path="cag_model",
        python_model=CAGApplianceBot(SYSTEM_PROMPT, "databricks-meta-llama-3-3-70b-instruct"),
        registered_model_name=registered_model_name,
        signature=signature,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "databricks-langchain",    # ✅ Critical: adds the dedicated integration
            "langchain",
            "langchain-core",
            "databricks-sdk"
        ]
    )

print(f"New version logged and registered to {registered_model_name}.")