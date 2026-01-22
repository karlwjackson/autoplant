
import json
import azure.functions as func

app = func.FunctionApp()

@app.function_name(name="HttpIngest")
@app.route(route="ingest", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
@app.blob_output(arg_name="outputBlob", path="ingest/{sys.utcnow}.json", connection="AzureWebJobsStorage")
def run(req: func.HttpRequest, outputBlob: func.Out[str]) -> func.HttpResponse:
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse("Invalid JSON", status_code=400)

    outputBlob.set(json.dumps(body, ensure_ascii=False))
    return func.HttpResponse("Accepted", status_code=202)
