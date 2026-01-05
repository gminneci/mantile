from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .models import HardwareSpecs, ParallelismConfig, ModelIR
from .hardware_library import get_nvl72_specs, get_nvl72_rack_specs
from .ir_builder import build_model_ir
from .estimator import estimate_performance

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EstimateRequest(BaseModel):
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hardware_preset: str = "nvl72_rack"
    tp_size: int = 1
    batch_size: int = 1
    input_seq: int = 128
    output_seq: int = 128

@app.get("/hardware")
def list_hardware():
    return {
        "nvl72_single": get_nvl72_specs(),
        "nvl72_rack": get_nvl72_rack_specs()
    }

@app.post("/estimate")
def run_estimation(req: EstimateRequest):
    # 1. Load Hardware
    if req.hardware_preset == "nvl72_rack":
        hw = get_nvl72_rack_specs()
    elif req.hardware_preset == "nvl72_single":
        hw = get_nvl72_specs()
    else:
        raise HTTPException(status_code=404, detail="Hardware preset not found")

    # 2. Build IR
    try:
        # Generic builder works for any transformer architecture
        ir = build_model_ir(req.model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
    # 3. Parallel Config
    par = ParallelismConfig(
        tp_size=req.tp_size,
        batch_size=req.batch_size,
        input_seq_len=req.input_seq,
        output_seq_len=req.output_seq
    )
    
    # 4. Estimate
    result = estimate_performance(hw, ir, par)
    
    return result
