import streamlit as st
import torch

# Lazy import transformers to avoid top-level failure
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.error(f"Transformers import failed: {e}. Check logs.")
    TRANSFORMERS_AVAILABLE = False
    st.stop()

from peft import PeftModel
import re

if not TRANSFORMERS_AVAILABLE:
    st.stop()

# Load model (Mistral-7B: fast, ungated, story-perfect)
@st.cache_resource(ttl=3600)
def load_model():
    base_model = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    )
    # Optional LoRA (comment out if slow; uses public story adapter)
    # model = PeftModel.from_pretrained(model, "SartajBhuvaji/storyforge-lora")
    return model, tokenizer

model, tokenizer = load_model()

# Emotion classifier
@st.cache_resource
def load_emotion_clf():
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=3)

emotion_clf = load_emotion_clf()

st.set_page_config(page_title="StoryForge by Harshil2498", layout="wide")
st.title("🖋️ StoryForge")
st.caption("Emotion-adaptive multilingual story co-writer. Built by Harshil2498 🚀")

# Sidebar
with st.sidebar:
    mood = st.select_slider("Starting mood", options=["auto", "joy", "sadness", "anger", "fear", "love"])
    lang = st.selectbox("Language", ["English", "Español", "Français", "हिंदी"])

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a masterful storyteller. Adapt to the user's emotion and language."}]

# Display chat
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if prompt := st.chat_input("Start your story... (e.g., 'I feel lost in the rain')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # Detect emotion
        if mood == "auto":
            emotions = emotion_clf(prompt)
            detected = emotions[0]['label']
            st.caption(f"Detected mood: **{detected}**")
        else:
            detected = mood

        # Build prompt
        system_prompt = f"Write an emotional story continuation. Mood: {detected}. Language: {lang}. Match user's style."
        full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.8, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.caption("💡 Built with Streamlit + Mistral-7B. GitHub: [Harshil2498/StoryForge](https://github.com/Harshil2498/StoryForge)")
