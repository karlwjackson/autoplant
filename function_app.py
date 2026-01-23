
import azure.functions as func
import json
import logging
from azure.storage.blob import BlobServiceClient
import os
import uuid

app = func.FunctionApp()

@app.function_name(name="HttpIngest")
@app.route(route="ingest", methods=["POST"])
def ingest(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("HttpIngest triggered")

    # Parse JSON from the Arduino
    try:
        data = req.get_json()
    except:
        return func.HttpResponse("Invalid JSON", status_code=400)

    # Write JSON to Blob Storage
    conn = os.getenv("AzureWebJobsStorage")
    blob_name = f"{uuid.uuid4()}.json"

    bsc = BlobServiceClient.from_connection_string(conn)
    container = bsc.get_container_client("ingest")
    container.upload_blob(blob_name, json.dumps(data), overwrite=True)

    return func.HttpResponse("OK", status_code=202)
