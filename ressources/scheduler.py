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
}

def get_scheduler(pipe, scheduler):
    if scheduler in SCHEDULER_MAPPING:
        SchedulerClass = SCHEDULER_MAPPING[scheduler]
        pipe.scheduler = SchedulerClass.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Invalid scheduler name {scheduler}")

    return pipe
