!pip install -q torchaudio
!pip install torch
!pip install -q pydub
!pip3 install -q -U bitsandbytes==0.42.0
!pip3 install -q -U peft==0.8.2
!pip3 install -q -U trl
!pip3 install -q -U accelerate==0.27.1
!pip3 install -q -U datasets==2.17.0
!pip3 install -q -U transformers==4.38.0
USE_ONNX = False # change this to True if you want to test onnx model
if USE_ONNX:
    !pip install -q onnxruntime
!pip install -q huggingsound
#@title Install and Import Dependencies

# this assumes that you have a relevant version of PyTorch installed
SAMPLING_RATE = 16000
import torch
torch.set_num_threads(1)
from IPython.display import Audio
from pprint import pprint
# download example
# torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import re
import json
import torch
import torch.nn as n
import transformers
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
from datasets import Dataset
from peft import LoraConfig, PeftConfig
import bitsandbytes as bnb
from trl import SFTTrainer
from tqdm.notebook import tqdm
# from sklearn.metrics import (accuracy_score,
#                              classification_report,
#                              confusion_matrix)
# from sklearn.model_selection import train_test_split
from datasets import load_dataset
import os
os.environ["WANDB_DISABLED"] = "true"


model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=USE_ONNX)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

model_name = "model/qwen2-72b"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model_txt = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    local_files_only = True
)

model_txt.config.use_cache = False
model_txt.config.pretraining_tp = 1

tokenizer_txt = AutoTokenizer.from_pretrained(model_name)
## using VADIterator class
# ! ffmpeg -i /content/voice_11318.mp4 -acodec pcm_s16le -ar 16000 -y output.wav

from pydub import AudioSegment
import os
from huggingsound import SpeechRecognitionModel
import torch
import transformers
vad_iterator = VADIterator(model)
timestamps = []
wav = read_audio(f'/content/output.wav', sampling_rate=SAMPLING_RATE)

window_size_samples = 1536 # number of samples in a single audio chunk
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+ window_size_samples]
    if len(chunk) < window_size_samples:
      break
    speech_dict = vad_iterator(chunk, return_seconds=True)
    if speech_dict:
        timestamps.append(speech_dict)
        # print(speech_dict, end=' ')
vad_iterator.reset_states() # reset model states after each audio


def cut_wav_chunks(input_file, timestamps):
    # Load the audio file
    audio = AudioSegment.from_wav(input_file)

    chunks = []
    for i in range(0, len(timestamps), 2):
        start = timestamps[i].get('start', 0) * 1000  # Convert to milliseconds
        end = timestamps[i+1].get('end', len(audio)) * 1000  # Convert to milliseconds
        chunk = audio[start:end]
        chunks.append(chunk)

    return chunks
# Path to your input WAV file
input_file = "/content/output.wav"
output_dir = "/content/sound"
os.makedirs(output_dir, exist_ok=True)
# Cut the chunks
chunks = cut_wav_chunks(input_file, timestamps)

# Save the chunks
for i, chunk in enumerate(chunks):

    chunk.export(f"/content/sound/chunk_{i+1}.wav", format="wav")

# Function to extract the numeric value from the filename


audio_path = []
for a in os.listdir('/content/sound'):
  full_path = os.path.join('/content/sound', a)
  if os.path.isfile(full_path):
    audio_path.append(full_path)


def extract_number(file_path):
    filename = file_path.split('/')[-1]
    number = int(filename.split('_')[1].split('.')[0])
    return number

sorted_file_paths = sorted(audio_path, key=extract_number)



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = SpeechRecognitionModel("airesearch/wav2vec2-large-xlsr-53-th", device = device)

transcriptions = model.transcribe(sorted_file_paths)
chunk_word = []
full_transcript =''
for item in transcriptions:
    chunk_word.append(item['transcription'].replace(" ",""))
    print("//",item['transcription'].replace(" ",""),"//")
    full_transcript += ''.join(item['transcription'].replace(" ",""))


def extract_and_combine(text):
    """
    Extract JSON strings between |<start>| and |<end>| markers and combine them into a single JSON array.

    :param text: The input text containing the JSON strings
    :return: A combined JSON string
    """
    trl_ar = []
    # Regular expression to find JSON strings between |<start>| and |<end>|
    pattern = r'\|<start>\|(.*?)\|<end>\|'
    matches = re.findall(pattern, text)
    [trl_ar.append(matche) for matche in matches]
    # # Parse the matched JSON strings and combine them into a list
    # json_objects = [json.loads(match) for match in matches]
    
    # # Convert the list of JSON objects back to a JSON string
    # combined_json_string = json.dumps(json_objects, ensure_ascii=False)
    
    return trl_ar

def cutting(prompt):
    messagee = prompt.split(" ")
    return messagee


tokenizer.pad_token_id = tokenizer.eos_token_id
pipeline = transformers.pipeline(
    "text-generation", model=model_txt,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.float16},
    trust_remote_code = True,
    device_map="auto",
)

if full_transcript:
    prompt = full_transcript
else:
    prompt = """มีการรายงานข่าวเกี่ยวกับเหตุการณ์ที่เจ้าหน้าที่ตำรวจเข้าตรวจสอบอาจารย์หญิงคนหนึ่งที่เป็นผู้นำทางจิตวิญญาณของชาวบ้านในสำนักสุขาวดี จังหวัดอุดรธานี โดยตอนแรกเจ้าหน้าที่ไม่ทราบชื่อจริงของเธอ จนกระทั่งพบว่าเธอเคยเปลี่ยนชื่อมาหลายครั้ง และมีหมายจับเกี่ยวกับการทุจริตและการนำเข้าข้อมูลอันเป็นเท็จสู่ระบบคอมพิวเตอร์เมื่อเจ้าหน้าที่ควบคุมตัวอาจารย์หญิงคนนี้ เธอกล่าวว่าไม่ได้รู้สึกเครียดและบอกว่าร่างของเธอถูกครอบครองโดยวิญญาณอื่น จากการตรวจสอบพบว่าเธอเคยเป็นนักธุรกิจและซีอีโอของบริษัทหนึ่ง ซึ่งหลังจากมีคดีความเกี่ยวกับการเงิน เธอก็หันมาเป็นผู้นำทางจิตวิญญาณในห้องขัง เธอนั่งสมาธิและอ้างว่าได้ช่วยปลดปล่อยดวงวิญญาณนักรบในอดีตหลายหมื่นดวง อย่างไรก็ตาม เจ้าหน้าที่ตำรวจยืนยันว่าเธอมีคดีความที่ยังต้องสอบสวนและมีการควบคุมตัวต่อไป ซึ่งเธอได้รับการประกันตัวในที่สุดเหตุการณ์นี้ทำให้หลายคนสงสัยเกี่ยวกับความเชื่อและพฤติกรรมของเธอ รวมถึงการดำเนินการของเจ้าหน้าที่ในการตรวจสอบคดีความของเธอที่เกี่ยวข้องกับการทุจริตในอดีต"""

messagee = cutting(prompt)
persona_llm = """you are Narin Phon an experrt in correcting sentence
Skills:
Text Correction: Expertise in identifying and correcting grammatical, syntactical, and contextual errors in Thai text.
Data Annotation: Experience in creating and managing annotated datasets for training language models."""
massive = []
for i in messagee:
    txt_messages = [
        {"role": "system", "content": f"""{persona_llm}.
        you must answer like this [
            input =  พระบรมมทาตสวีตั้งอยู่บนถนนเพชรเกษมที่ตั้งท่ามกลางทะเลอันดาหมั่นของจังหวัดสงขลานมีตำนานเกี่ยวกับอารังกาศักดิ์สิทธิิ์ซึ่งรวมอยู่ในภูเหารับล่อใกล้เคียงขับ
            output = |<start>|พระบรมธาตุสวีตั้งอยู่บนถนนเพชรเกษมที่ตั้งท่ามกลางทะเลอันดามันของจังหวัดสงขลา มีตำนานเกี่ยวกับลังกาศักดิ์สิทธิ์ซึ่งร่วมอยู่ในภูเขารับร่อใกล้เคียงครับ|<end>|
            ]
            if output has more than one use "," between them 
        """},
        {"role": "user", "content":f"{i}"}
    ]
    massive.append(txt_messages)

outputs = []
for j in tqdm(range(len(messagee))):
    i = messagee[j]
    txt_messages = massive[j]
    out = pipeline(
        txt_messages,
        max_new_tokens=128,
        temperature=0.01,
        early_stopping=True,
        num_beams=1,top_k=3, 
        eos_token_id=model.config.eos_token_id
    )
    # print(out)
    outputs.append(out)


cleaned = []
for i in outputs:
    output_text = extract_and_combine(i[0]["generated_text"][2]["content"])
    cleaned.append(output_text[0])
# df = to_dfa(cleaned)
# df
correct_txt = " ".join(cleaned)
correct_txt