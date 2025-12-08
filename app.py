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
    return pipeline(
        "text-generation",
        model="pranavpsv/gnome-1b-storywriter",   # ←←← THIS IS THE MAGIC
        torch_dtype=torch.float16,
        device_map="auto"
    )

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
            full_prompt = f"You are a poetic storyteller. Continue this story in {lang}, in a deeply {emotion_adj} and beautiful style. Output ONLY the story continuation, no introductions, no explanations:\n\n{prompt}"
            output = generator(full_prompt, max_new_tokens=80, min_new_tokens=40, temperature=0.85, do_sample=True, repetition_penalty=1.2, top_p=0.9)
            response = output[0]["generated_text"][len(full_prompt):].strip()  # Extract new text only

            # Fallback (if ultra-rare short)
            if len(response) < 20:
                response = f"In the {emotion_adj} rain, you wander lost, but a spark of {detected} ignites a new path—whispers of adventure calling through the storm."

            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.caption("💡 Powered by GPT-2 + DistilRoBERTa emotion detection. GitHub: [Harshil2498/StoryForge](https://github.com/Harshil2498/StoryForge)")
