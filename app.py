
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load fine-tuned model (store these files in the same directory as this script on Render)
model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to("cuda" if torch.cuda.is_available() else "cpu")

class Message(BaseModel):
    user_input: str

chat_history_ids = None

@app.post("/chat")
def chat(msg: Message):
    global chat_history_ids
    input_ids = tokenizer.encode(msg.user_input + tokenizer.eos_token, return_tensors='pt')
    input_ids = input_ids.to(model.device)

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
    else:
        bot_input_ids = input_ids

    chat_history_ids = model.generate(
        bot_input_ids, max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return {"response": response}