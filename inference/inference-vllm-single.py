
import time
import cv2
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import json


# Set model Path
MODEL_PATH  = ""   
# Set video path
VIDEO_PATH = "test.mp4"


NUM_FRAMES   = 16
MAX_PIXELS   = 128 * 28 * 28         
ALIGN_FACTOR = 28                   


BINARY_PROMPT = """Your task is to analyze whether the given video is abnormal or normal. Think before answering, and generate:
1. A structured reasoning process enclosed in <think></think> tags
2. A final explanation enclosed in <answer></answer> tags

For abnormal videos, the reasoning should be based on a structured 4-step process:
<think> must include the following four steps enclosed in corresponding tags:
<step1>: Scene Description — Provide an objective overview of the environment and normal behaviors, without mentioning any abnormal activity or speculation.
<step2>: Abnormal Event Description — Describe the abnormal event and its approximate spatial location (e.g., bottom left of the frame), without explaining why it is abnormal.
<step3>: Abnormal Event Recognition — Explain why this event is considered abnormal compared to normal patterns or expectations.
<step4>: Causal Reasoning and Social Norms — Analyze potential negative consequences and explain how this behavior violates social norms or expectations.

<answer> must be a single, coherent paragraph in natural language, which includes exactly the following five tags:
<which>: Define the video as \"Abnormal.\"
<what>: What happened (describe the anomalous event)
<when>: When it happened, in normalized frame indices (e.g., <when>[0.25, 0.45]</when>)
<where>: Where it happened (use approximate spatial descriptions)
<why>: Why it is considered abnormal
<how>: How this behavior could cause harm or violate norms

Example Output for Abnormal Videos:
<think> 
<step1>The video shows ...</step1>
<step2>Next, we observe an abnormal event ...</step2> 
<step3>Based on these observations ...</step3> 
<step4>As a result, this behavior ...</step4> 
</think> 
<answer>
The video is classified as <which>Abnormal</which>. In this video, <what>a pedestrian ...</what>, occurring during the time range <when>[0.121, 0.826]</when>. The event takes place approximately in the <where>lower-left area of the frame</where>. This is considered abnormal because <why>pedestrians are expected to ...</why>. As a result, <how>such behavior could ...</how>.
</answer>

For normal videos, the reasoning should be simplified to just two steps:
<think> must include only the following two steps:
<step1>: Scene and Object Description — Provide a concise and objective overview of the environment and typical behaviors, without mentioning anomalies.
<step2>: Normal Event Explanation — Explain why the video is considered normal.

<answer> must be a single, coherent paragraph in natural language, which includes the following three tags:
<which>: Define the video as \"Normal.\"
<what>: A concise description of the event in the video.
<why>: Why it is considered normal.

Example Output for Normal Videos:
<think> 
<step1>The video shows scenes ...</step1> 
<step2>Based on the described scenes ...</step2> 
</think> 
<answer>
The video is classified as <which>Normal</which>. In this video, <what>people are ...</what>. This is considered normal because <why>...</why>
</answer>
"""


SYSTEM_PROMPT = ("You are a multimodal reasoning assistant for "
                 "understanding anomalies in videos.")


def compress_frame(frame,
                   max_pixels: int = MAX_PIXELS,
                   factor: int = ALIGN_FACTOR):
    h, w = frame.shape[:2]

    if (h * w <= max_pixels) and (h % factor == 0) and (w % factor == 0):
        return frame

    scale = (max_pixels / (h * w)) ** 0.5
    new_h = max(int((h * scale) // factor) * factor, factor)
    new_w = max(int((w * scale) // factor) * factor, factor)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sample_video(path: str, num_frames: int = NUM_FRAMES):
 
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise RuntimeError(f"Cannot read video: {path}")

    idxs = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = compress_frame(frame)
            frames.append(frame)

    cap.release()
    if len(frames) != num_frames:
        print(f"Warning: {len(frames)} - {num_frames} ")
        # raise RuntimeError("Frame sampling failed.")
    return np.stack(frames)      # shape (T, H, W, C)


llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    dtype="bfloat16",                 
    max_model_len=8192,
    mm_processor_kwargs={
        "min_pixels": ALIGN_FACTOR * ALIGN_FACTOR,
        "max_pixels": MAX_PIXELS,
        "fps": 1,                 
    },
    limit_mm_per_prompt={"video": 1},
    gpu_memory_utilization=0.8,    
)

tokenizer = AutoProcessor.from_pretrained(MODEL_PATH,
                                          trust_remote_code=True)


messages = [

    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": [
            {
                "type":   "video",
                "video":  "dummy_path.mp4",  
                "nframes": NUM_FRAMES,        
            },
            {
                "type":  "text",
                "text":  BINARY_PROMPT
            }
        ]
    }
]

prompt = tokenizer.apply_chat_template(messages,
                                       tokenize=False,
                                       add_generation_prompt=True)



video_ndarray = sample_video(VIDEO_PATH, NUM_FRAMES)

inputs = [{
    "prompt": prompt,
    "multi_modal_data": {
        "video": video_ndarray
    }
}]

sampling = SamplingParams(
    temperature=0.1,
    top_p=0.9,
    max_tokens=512,
)

t0 = time.time()
result = llm.generate(inputs, sampling_params=sampling)
t1 = time.time()


print("\n=== Generation Result ===\n")
print(result[0].outputs[0].text)
print(f"\nTime cost: {t1 - t0:.2f} s")
