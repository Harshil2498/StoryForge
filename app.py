import streamlit as st
import torch

try:
    from transformers import pipeline
except ImportError as e:
    st.error(f"Transformers import failed: {e}")
    st.stop()

st.set_page_config(page_title="StoryForge by Harshil2498", layout="wide")
st.title("🖋️ StoryForge")
st.caption("Emotion-adaptive multilingual story co-writer. Built by Harshil2498 in 1 day 🚀")

# Emotion classifier (fast)
@st.cache_resource
def load_emotion_clf():
    try:
        return pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        top_k=3)
    except Exception as e:
        st.error(f"Emotion model failed: {e}. Using manual mood.")
        return None

emotion_clf = load_emotion_clf()

# GPT-2 pipeline (simple, always generates)
# Tiny, beautiful story model (loads in 1 second, never rambles)
@st.cache_resource
def load_generator():
    try:
        model_name = "HuggingFaceH4/zephyr-7b-alpha"
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True  # Required for MPT
        )
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Model load failed: {e}. Falling back to simple mode.")
        return pipeline("text-generation", model="gpt2")  # Backup

generator = load_generator()

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
        with st.spinner("Crafting your story... (2-5s)"):
            # Detect emotion
            detected = mood if mood != "auto" else "neutral"
            if emotion_clf:
                try:
                    result = emotion_clf(prompt)[0]
                    detected = result[0]["label"].lower()
                    st.caption(f"Detected emotion: **{detected}**")
                except:
                    pass

            # Build enhanced prompt for GPT-2 (forces story)
            lang_map = {"English": "", "Español": "Escribe en español: ", "Français": "Écrivez en français: ", "हिंदी": "हिंदी में लिखें: ", "Deutsch": "Schreiben Sie auf Deutsch: "}
            emotion_map = {"sadness": "heartbreaking", "anger": "furious", "joy": "joyful", "fear": "terrifying", "love": "romantic", "surprise": "unexpected", "neutral": "mysterious"}
            emotion_adj = emotion_map.get(detected, "mysterious")
            full_prompt = f"""<|system|>
             You are a poetic storyteller. 
                Write ONLY a short, beautiful continuation of the user's story.
            Emotion: {detected}
            Language: {lang}
            Never add introductions, explanations, or extra text.
            </|system|>
            <|user|>
            {prompt}
            </|user|>
            <|assistant|>"""

           

            output = generator(full_prompt, max_new_tokens=120, min_new_tokens=50, temperature=0.8, do_sample=True, repetition_penalty=1.05)
            response = output[0]["generated_text"][len(full_prompt):].strip()

            if len(response) < 30:
               response = f"The rain envelops you in its gentle {detected} embrace, washing away the edges of your sorrow. In the downpour, a quiet revelation blooms: loss is but the soil for new growth."
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.caption("💡 Powered by GPT-2 + DistilRoBERTa emotion detection. GitHub: [Harshil2498/StoryForge](https://github.com/Harshil2498/StoryForge)")
