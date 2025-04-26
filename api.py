import os
import sys
import uvicorn
import time
import asyncio
from loguru import logger
from PIL import Image
import numpy as np
from sse_starlette.sse import EventSourceResponse
import json

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Import Fooocus modules - correct import order
import modules.config as config
import modules.flags as flags
from modules.sdxl_styles import legal_style_names
import modules.api_async_worker as async_worker
from modules.api_params import GenerateParams
import args_manager

# Ensure outputs directory exists
outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)
app.state.current_task = None

# Fix static files configuration
app.mount("/outputs", StaticFiles(directory=outputs_dir, html=True))

# API Endpoints
@app.get("/api/styles")  
async def get_styles():
    logger.info("Fetching styles")
    try:
        return {
            "styles": legal_style_names,
            "default_styles": config.default_styles
        }
    except Exception as e:
        logger.error(f"Error getting styles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_models():
    logger.info("Fetching models")
    try:
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        base_dir = os.path.join(models_dir, 'checkpoints')
        lora_dir = os.path.join(models_dir, 'loras')
        
        # Scan for models like webui.py does
        base_models = [f for f in os.listdir(base_dir) if f.endswith(('.safetensors', '.ckpt'))]
        refiner_models = ["None"] + [f for f in base_models if 'refiner' in f.lower()]
        loras = [f for f in os.listdir(lora_dir) if f.endswith(('.safetensors', '.ckpt'))]
        
        return {
            "base_models": base_models,
            "refiner_models": refiner_models,
            "loras": loras
        }
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/configs")
async def get_configs():
    logger.info("Fetching configs")
    try:
        return {
            "performance_selections": flags.Performance.values(),
            "aspect_ratios": config.available_aspect_ratios,
            "uov_methods": flags.uov_list,
            "default_settings": {
                "guidance_scale": config.default_cfg_scale,
                "sharpness": config.default_sample_sharpness,
                "performance": config.default_performance,
                "negative_prompt": config.default_prompt_negative,
                "style_selections": config.default_styles
            }
        }
    except Exception as e:
        logger.error(f"Error getting configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_clicked():
    """Stream generation results using SSE."""
    execution_start_time = time.perf_counter()
    task = app.state.current_task  # Get the current task
    async_worker.async_tasks.append(task)

    finished = False
    
    try:
        while not finished:
            await asyncio.sleep(0.01)
                
            if task.yields:
                flag, product = task.yields.pop(0)
                message = None
                
                if flag == 'preview':
                    percentage, title, _ = product  # Ignore the image data
                    logger.info(f"Preview event: {percentage}% - {title}")
                    preview_data = {
                        "percentage": percentage,
                        "title": title
                    }
                    message = {"event": "preview", "data": preview_data}
                        
                elif flag == 'results':
                    if isinstance(product, list):
                        logger.info("Processing results event")
                        image_urls = await save_results(product)
                        message = {"event": "results", "data": {"images": image_urls}}
                        
                elif flag == 'finish':
                    if isinstance(product, list):
                        logger.info("Processing finish event")
                        final_urls = await save_results(product)
                        message = {
                            "event": "finish", 
                            "data": {
                                "images": final_urls,
                                "execution_time": time.perf_counter() - execution_start_time
                            }
                        }
                    finished = True

                if message:
                    logger.debug(f"Sending SSE message: {message}")
                    yield f"data: {json.dumps(message)}\n\n"
                        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        message = json.dumps({"error": str(e)})
        yield f"data: {message}\n\n"
    finally:
        app.state.current_task = None  # Clear the current task
        if task in async_worker.async_tasks:
            async_worker.async_tasks.remove(task)
            logger.info("Task removed from active tasks")

async def save_results(images):
    """Save result images and return URLs"""
    image_urls = []
    timestamp = int(time.time())
    
    for idx, img in enumerate(images):
        filename = f"result_{timestamp}_{idx}.png"
        filepath = os.path.join(outputs_dir, filename)  # 使用完整路径
        
        if isinstance(img, str) and os.path.exists(img):
            with Image.open(img) as source_img:
                source_img.save(filepath)
        elif isinstance(img, np.ndarray):
            Image.fromarray(img).save(filepath)
            
        image_urls.append(f"/outputs/{filename}")
        
    return image_urls

@app.post("/api/create_task")
async def create_task(reqParams: GenerateParams):
    """Create a new AsyncTask and set it as the current task."""
    try:
        os.makedirs("outputs", exist_ok=True)
        task = async_worker.AsyncTask(reqParams)
        app.state.current_task = task  # Save the task as the current task
        logger.info("Task created and set as the current task")
        return JSONResponse(content={"status": "ok"})
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/generate")
async def generate_images():
    """Generate images for the current task using SSE."""
    try:
        logger.info("Received request to generate images for the current task")
        
        task = app.state.current_task
        if task is None:
            logger.error("No current task found")
            raise HTTPException(status_code=404, detail="No task found")

        logger.info("Starting generation for the current task")
        return EventSourceResponse(generate_clicked())
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7866)
