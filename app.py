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

# Quantized model
@st.cache_resource
def load_model():
    try:
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
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
        with st.spinner("Detecting emotion & crafting story... (5-10s)"):
            # Detect emotion
            detected = mood if mood != "auto" else "neutral"
            if emotion_clf:
                try:
                    result = emotion_clf(prompt)[0]
                    detected = result[0]["label"].lower()
                    st.caption(f"Detected emotion: **{detected}**")
                except:
                    pass

            # Load & generate
            model, tokenizer = load_model()
            system = f"You are a masterful storyteller. Write in {lang}. Infuse with {detected} emotion. Match user's poetic style. Keep it concise (200 words max)."
            full_prompt = f"<s>[INST] {system} \n\n{prompt} [/INST]"

            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            input_length = inputs['input_ids'].shape[1]
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    min_new_tokens=50,  # **FIX: Forces minimum output length**
                    temperature=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.05  # **FIX: Lower to avoid over-penalizing**
                )
            full_response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            
            # **FIX: Robust extraction**
            response = full_response.split("[/INST]")[-1].strip() if "[/INST]" in full_response else full_response[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
            
            # Debug (remove after testing)
            new_tokens = len(output[0]) - input_length
            st.caption(f"**Debug:** Generated {new_tokens} tokens")  # Should be >50

            if len(response) < 20:  # **FIX: Fallback if still short/empty**
                fallback_stories = {
                    "sadness": "The rain falls like a veil of forgotten promises, each drop tracing paths down your skin like tears you can't shed. You wander cobblestone streets, the city's hum a distant lullaby, until a flickering lantern draws you to a hidden bookstore. Inside, pages whisper secrets of wanderers who found home in storms just like yours.",
                    "anger": "The rain lashes like whips of fury, mirroring the storm raging in your chest. You clench your fists, puddles splashing under defiant steps, vowing to shatter the chains of this endless downpour. A thunderclap echoes your roar—power surges, turning despair to rebellion.",
                    "neutral": "The rain patters softly, a rhythmic companion to your thoughts. Lost in its melody, you notice the world sharpening: leaves glistening, lights reflecting in puddles like stars fallen to earth. Step by step, the path unfolds, leading you toward clarity."
                }
                response = fallback_stories.get(detected, "In the heart of the storm, a quiet voice emerges: 'This too shall pass.' The rain eases, revealing a world renewed, ready for your next chapter.")
                st.caption("**Fallback activated:** Using emotion-inspired default story.")

            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.caption("💡 Powered by quantized Mistral-7B + DistilRoBERTa emotion detection. GitHub: [Harshil2498/StoryForge](https://github.com/Harshil2498/StoryForge)")
