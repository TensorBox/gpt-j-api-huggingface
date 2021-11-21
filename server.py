from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from inference import run_inference
import uvicorn

app = FastAPI()

@app.post("/generate")
async def root(request: Request):
    params_json = await request.json()
    
    if "prompt" not in params_json:
        raise HTTPException(status_code=400, detail="Prompt needs to provided as an input parameter")

    output = run_inference(params_json)

    return {"output": output}

uvicorn.run(app, host="0.0.0.0", port=5000)