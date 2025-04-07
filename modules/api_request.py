from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from modules.flags import (
    Performance,
)


# Request Models
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    style_selections: List[str] = []  # 修改为匹配AsyncTask的字段名
    performance_selection: Performance = Field(default=Performance.SPEED, description="Performance selection")
    aspect_ratios_selection: str = Field(default="1152×896")
    image_number: int = 1
    seed: int = -1
    sharpness: float = 2.0
    cfg_scale: float = 4.0  # 修改名称从guidance_scale为cfg_scale
    base_model_name: str = "juggernautXL_version6Rundiffusion.safetensors"
    refiner_model_name: str = "None"
    refiner_switch: float = 0.5
    loras: List[Dict] = []
    input_image: Optional[str] = None
    uov_method: Optional[str] = None

    # Advanced Parameters
    clip_skip: int = 1
    sampler: str = "dpmpp_2m"
    scheduler: str = "karras"
    vae: str = 'Default (model)'
    
    # Performance Parameters
    performance: str = "Speed"
    
    # Image Settings
    resolution: str = "1024×1024"
    
    # Style Settings
    styles: List[str] = []
    
    # ControlNet Parameters
    controlnet_softness: float = 0.25
    canny_low_threshold: float = 0.3
    canny_high_threshold: float = 0.7
    
    # FreeU Parameters
    freeu_enabled: bool = False
    freeu_b1: float = 1.1
    freeu_b2: float = 1.2
    freeu_s1: float = 0.9
    freeu_s2: float = 0.2
    
    # Inpainting Parameters
    inpaint_engine: str = "v1"
    inpaint_strength: float = 1.0
    inpaint_respective_field: float = 1.0
    
    # Enhancement Parameters
    enhance_prompt: str = ""
    enhance_negative_prompt: str = ""
    enhance_steps: int = 0
    enhance_uov_method: str = "disabled"
    
    # Metadata
    save_metadata: bool = True
    metadata_scheme: str = "FOOOCUS"

    class Config:
        extra = "allow"  # Allow additional fields for future compatibility

    @validator('aspect_ratios_selection')
    def validate_aspect_ratio(cls, v):
        if not v or '×' not in v:
            return "1024×1024"
        try:
            w, h = v.replace('×', ' ').split(' ')[:2]
            w, h = int(w), int(h)
            return f"{w}×{h}"
        except:
            return "1024×1024"