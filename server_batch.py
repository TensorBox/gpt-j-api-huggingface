from fastapi import FastAPI, Request, HTTPException
from inference_batch import InferenceModel
import asyncio

MAX_TOKENS = 4778 # HACK: determined the maximum amount of tokens for Nvidia T4 (16 Gb VRAM) 
                  # by manually tuning the number of items in a batch with items of sequence length of 1024
MAX_QUEUE_SIZE = 20
DEFAULT_MAX_SEQ_LEN = 20

inference_model = InferenceModel()

app = FastAPI()

q = asyncio.Queue(MAX_QUEUE_SIZE)

async def process_batches():
    first_item = None
    while True:
        if first_item is None:
            first_item = await q.get() # wait for the next item (blocking)
        items = [first_item]
        batch_max_len = first_item[0]['max_length']
        
        # dequeue until we get an exception which means that the queue is empty
        while True: 
            q_item = None
            try:
                q_item = q.get_nowait()
            except Exception as e:
                print("Exception while getting the future", e)
                first_item = None
                break
            if q_item is not None:
                curr_max_len = max(batch_max_len, q_item[0]['max_length']) 
                if curr_max_len * len(items) < MAX_TOKENS:
                    items.append(q_item)
                    batch_max_len = curr_max_len
                else:
                    # Since adding this item to the batch will produce a batch that is too large to fit into a memory,
                    # we use it as the first item during the next iteration.
                    first_item = q_item
                    break
        
        outputs = inference_model.run_batch_inference([item[0] for item in items])

        # resolving the futures (setting the results)
        for i, output in enumerate(outputs):
            items[i][1].set_result(output)
        # HACK: the following line ensures that the futures are resolved 
        # before moving to processing next items
        await asyncio.sleep(0)        
        
loop = asyncio.get_event_loop()
task = loop.create_task(process_batches())

@app.post("/generate")
async def root(request: Request):
    params_json = await request.json()
    
    if "prompt" not in params_json:
        raise HTTPException(status_code=400, detail="Prompt needs to provided as an input parameter")
    
    if "max_length" not in params_json:
        params_json["max_length"] = DEFAULT_MAX_SEQ_LEN
    
    fut = loop.create_future()
    try:
        q.put_nowait((params_json, fut))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Maximum amount of requests has been reached. Try again later")
    print(q.qsize())
    res = await fut
    return res

from uvicorn import Config, Server
config = Config(app=app, loop=loop, host="0.0.0.0", port=5000)
server = Server(config)
loop.run_until_complete(server.serve())
