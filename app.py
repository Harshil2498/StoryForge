import streamlit as st
import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
except ImportError as e:
    st.error(f"Transformers import failed: {e}")
    st.stop()

st.set_page_config(page_title="StoryForge by Harshil2498", layout="wide")
st.title("🖋️ StoryForge")
st.caption("Emotion-adaptive multilingual story co-writer. Built by Harshil2498 in 1 day 🚀")

# Emotion classifier
@st.cache_resource
def load_emotion_clf():
    try:
        return pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        top_k=3, device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        st.error(f"Emotion model load failed: {e}. Using manual mood.")
        return None

emotion_clf = load_emotion_clf()

# Phi-3 Mini model (faster, reliable)
@st.cache_resource
def load_model():
    try:
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True  # Required for Phi-3
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Story model load failed: {e}. App restarting...")
        st.rerun()

# Sidebar
with st.sidebar:
    mood = st.select_slider("Starting mood", options=["auto", "joy", "sadness", "anger", "fear", "love", "surprise"])
    lang = st.selectbox("Language", ["English", "Español", "Français", "हिन्दी", "Deutsch"])

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
        with st.spinner("Detecting emotion & crafting story... (3-8s)"):
            # Detect emotion
            detected = mood if mood != "auto" else "neutral"
            if emotion_clf:
                try:
                    result = emotion_clf(prompt)[0]
                    detected = result[0]["label"].lower()
                    st.caption(f"Detected emotion: **{detected}**")
                except:
                    pass

            # Prepare messages for Phi-3 pipeline (fixes empty)
            model, tokenizer = load_model()
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            messages = [
                {"role": "system", "content": f"You are a masterful storyteller. Write in {lang}. Infuse with {detected} emotion. Keep concise (150 words max). Match user's style."},
                {"role": "user", "content": prompt}
            ]
            generation_args = {
                "max_new_tokens": 150,
                "min_new_tokens": 50,
                "temperature": 0.9,
                "do_sample": True,
                "return_full_text": False,  # Key: Only new text
                "repetition_penalty": 1.05
            }

            try:
                output = pipe(messages, **generation_args)
                response = output[0]["generated_text"].strip()
            except:
                response = ""  # Fallback trigger

            if len(response) < 20:
                fallback = {
                    "sadness": "The rain falls gently, each drop a whisper of sorrow echoing your lost heart. Streets blur into gray memories, but in the mist, a faint light emerges—perhaps a sign that even in loss, new paths await discovery.",
                    "anger": "The rain pounds like fists of fury, fueling the storm within you. You splash through puddles, rage building with every step, ready to confront the shadows that haunt. Strength rises from the chaos, forging a fiercer you."
                }
                response = fallback.get(detected, "In the gentle rain, a sense of calm washes over, guiding lost souls toward hidden wonders. The world renews, offering fresh beginnings in every drop.")

            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.caption("💡 Powered by Phi-3-Mini + DistilRoBERTa emotion detection. GitHub: [Harshil2498/StoryForge](https://github.com/Harshil2498/StoryForge)")
