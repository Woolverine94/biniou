# https://github.com/Woolverine94/biniou
#scheduler.py
from diffusers import (
    UniPCMultistepScheduler,
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    DEISMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    LCMScheduler,
    EDMDPMSolverMultistepScheduler,
    EDMEulerScheduler,
    TCDScheduler,
    FlowMatchEulerDiscreteScheduler,
)

SCHEDULER_MAPPING = {
    "UniPC": UniPCMultistepScheduler,
    "DDIM": DDIMScheduler,
    "DDPM": DDPMScheduler,
    "PNDM": PNDMScheduler,
    "DEIS": DEISMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
    "Euler a": EulerAncestralDiscreteScheduler,
    "DPM2": KDPM2DiscreteScheduler,
    "DPM2 a": KDPM2AncestralDiscreteScheduler,
    "DPM++ SDE": DPMSolverSinglestepScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "Heun":	HeunDiscreteScheduler,
    "LMS": LMSDiscreteScheduler,
    "LCM": LCMScheduler,
    "DPM++ 2M Karras": DPMSolverMultistepScheduler,
    "DPM++ 2M SDE": DPMSolverMultistepScheduler,
    "DPM++ 2M SDE Karras": DPMSolverMultistepScheduler,
    "DPM++ SDE Karras": DPMSolverSinglestepScheduler,
    "DPM2 Karras": KDPM2DiscreteScheduler,
    "DPM2 a Karras": KDPM2AncestralDiscreteScheduler,
    "LMS Karras": LMSDiscreteScheduler,
    "EDM DPM++ 2M": EDMDPMSolverMultistepScheduler,
    "EDM Euler": EDMEulerScheduler,
    "TCD": TCDScheduler,
    "Flow Match Euler": FlowMatchEulerDiscreteScheduler,
}

SCHEDULER_MAPPING_MUSICLDM = {
    "DDIM": DDIMScheduler,
    "LMS": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

def get_scheduler(pipe, scheduler, **kwargs):
    if scheduler in SCHEDULER_MAPPING:
        SchedulerClass = SCHEDULER_MAPPING[scheduler]
        pipe.scheduler = SchedulerClass.from_config(pipe.scheduler.config, **kwargs)
    else:
        raise ValueError(f"Invalid scheduler name {scheduler}")
    return pipe
