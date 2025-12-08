import streamlit as st
import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError as e:
    st.error(f"Transformers import failed: {e}")
    st.stop()

st.set_page_config(page_title="StoryForge by Harshil2498", layout="wide")
st.title("🖋️ StoryForge")
st.caption("Emotion-adaptive multilingual story co-writer. Built by Harshil2498 in 1 day 🚀")

# Emotion classifier (reliable HF model)
@st.cache_resource
def load_emotion_clf():
    try:
        return pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        top_k=3, device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        st.error(f"Emotion model load failed (rare): {e}. Using manual mood.")
        return None

emotion_clf = load_emotion_clf()

# Mistral model (fast, ungated)
@st.cache_resource
def load_model():
    try:
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
    except Exception as e:
        st.error(f"Story model load failed: {e}. App restarting...")
        st.rerun()

# Sidebar
with st.sidebar:
    mood = st.select_slider("Starting mood", options=["auto", "joy", "sadness", "anger", "fear", "love", "surprise"])
    lang = st.selectbox("Language", ["English", "Español", "Français", "हिंदी", "Deutsch"])

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if prompt := st.chat_input("Start your story... (e.g., 'I feel lost in the rain')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Detecting emotion & crafting story..."):
            # Detect emotion
            detected = mood if mood != "auto" else "neutral"
            if emotion_clf:
                try:
                    result = emotion_clf(prompt)[0]
                    detected = result[0]["label"].lower()
                    st.caption(f"Detected emotion: **{detected}**")
                except:
                    pass  # Fallback to manual

            # Load model & generate
            model, tokenizer = load_model()
            system = f"You are a masterful storyteller. Write in {lang}. Infuse with {detected} emotion. Match user's poetic style."
            full_prompt = f"<s>[INST] {system} \n\n{prompt} [/INST]"

            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            output = model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.caption("💡 Powered by Mistral-7B + DistilRoBERTa emotion detection. GitHub: [Harshil2498/StoryForge](https://github.com/Harshil2498/StoryForge)")
