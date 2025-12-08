import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Tiny emotion classifier (loads instantly)
@st.cache_resource
def load_emotion_clf():
    return pipeline("text-classification", 
                    model="michelina/emotion-english-distilroberta-base", 
                    top_k=3)

emotion_clf = load_emotion_clf()

# Load Mistral-7B-Instruct (ungated, fast, excellent at stories)
@st.cache_resource
def load_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    return model, tokenizer

st.title("StoryForge by Harshil2498")
st.caption("Emotion-adaptive multilingual story co-writer • Built in 1 day")

with st.sidebar:
    mood = st.selectbox("Mood", ["auto", "joy", "sadness", "anger", "fear", "love"])
    lang = st.selectbox("Language", ["English", "Español", "Français", "हिंदी", "Deutsch"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Start or continue your story..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Feeling your emotion..."):
            # Detect emotion
            if mood == "auto":
                result = emotion_clf(prompt)[0]
                detected = result[0]["label"].lower()
                st.caption(f"Detected emotion: **{detected}**")
            else:
                detected = mood.lower()

            # Build prompt
            system = f"You are a brilliant storyteller. Write in {lang}. Current emotion: {detected}. Continue the story beautifully."
            full_prompt = f"<s>[INST] {system}\n\n{prompt} [/INST]"

            model, tokenizer = load_model()
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            output = model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            response = response.split("[/INST]")[-1].strip()

            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
