import streamlit as st
import os
import torch
from PIL import Image
from download_model import download_model

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline,
)

from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Image to Lyrics", layout="centered")
st.title("🎵 Image → Lyrics Generator")
st.markdown("""
<style>

/* Caption box */
.caption-box{
    background:#1e3a8a;
    padding:16px;
    border-radius:10px;
    margin-bottom:18px;
    color:white;
}

/* Emotion box */
.emotion-box{
    background:#6b21a8;
    padding:16px;
    border-radius:10px;
    margin-bottom:18px;
    color:white;
}

/* Lyrics box */
.lyrics-box{
    background:#0f172a;
    padding:18px;
    border-radius:10px;
    margin-bottom:18px;
    color:#f1f5f9;
    line-height:1.7;
    font-size:16px;
    white-space:pre-line;
}

/* Similarity box */
.similarity-box{
    background:#064e3b;
    padding:16px;
    border-radius:10px;
    color:#bbf7d0;
    font-weight:600;
}

/* Titles */
.box-title{
    font-size:20px;
    font-weight:600;
    margin-bottom:6px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# GROQ CONFIG
# -------------------------------------------------
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

client = Groq()

def generate_lyrics(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a creative lyricist."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=200,
    )

    return response.choices[0].message.content
# -------------------------------------------------
# EMOTION MODEL (GoEmotions)
# -------------------------------------------------

def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )
def detect_emotion(text):

    outputs = load_emotion_model()(text)[0]

    outputs_sorted = sorted(outputs, key=lambda x: x["score"], reverse=True)

    top = outputs_sorted[0]

    return top["label"]

# -------------------------------------------------
# BLIP IMAGE CAPTION (OFFLINE)
# -------------------------------------------------

#@st.cache_resource
def load_blip():

    model_path = download_model()

    import streamlit as st
    import os

    st.write("REAL model path:", model_path)
    st.write("Files in REAL path:", os.listdir(model_path))

    processor = BlipProcessor.from_pretrained(model_path)

    model = BlipForConditionalGeneration.from_pretrained(
        model_path
    ).eval()

    return processor, model

def generate_caption(image, processor, model):
    inputs = processor(image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        temperature=1.0,
        top_p=0.9,
        repetition_penalty=1.2
    )
    return processor.decode(output[0], skip_special_tokens=True)

# -------------------------------------------------
# SBERT SIMILARITY (FIXED)
# -------------------------------------------------

@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity(caption, lyrics, sbert):
    embeddings = sbert.encode([caption, lyrics])
    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]
    return float(score)

# -------------------------------------------------
# PROMPT BUILDER
# -------------------------------------------------
def build_prompt(caption, emotion, optional_text=None):
    return f"""
    You are an experienced songwriter.

    Transform the scenario into emotional song lyrics.

    Guidelines:

    * Let the emotion strongly shape the mood, imagery, and flow.
    * Use poetic language, soft metaphors, and sensory details.
    * Focus on feelings, atmosphere, and emotional storytelling rather than literal description.
    * Use simple, natural words that feel like real song lyrics.
    * Maintain a consistent emotional tone.

    Scenario:
    {caption}

    Dominant Emotion:
    {emotion}

    {f"Additional emotional direction: {optional_text}" if optional_text else ""}

    Output format:

    Write one long lyrical paragraph (6-8 flowing lines) describing the emotional scene.
    
    Requirements:

    * Do not label sections.
    * Do not explain the scene directly.
    * Avoid repeating the same phrases.
    * Focus on imagery, memories, and atmosphere.
    * Write only the lyrics.

    Generate expressive emotional lyrics.
    """


# -------------------------------------------------
# UI
# -------------------------------------------------
optional_text = st.text_input("",
    placeholder="Enter any additional context"
)

image_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if image_file:
    image = Image.open(image_file).convert("RGB")
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image(image, width=280)

    if st.button("Generate Lyrics 🎶"):

        # ✅ VERY IMPORTANT: define beforehand
        similarity_score = None

        with st.spinner("Running full AI pipeline..."):
            # Load models
            blip_p, blip_m = load_blip()
            
            sbert = load_sbert()

            # Caption
            caption = generate_caption(image, blip_p, blip_m)
            caption = caption.replace("< start >", "").replace("< end >", "").strip()
            
            # Emotion
            emotion = detect_emotion(caption+" "+optional_text)

            # Lyrics
            prompt = build_prompt(caption, emotion, optional_text)
            lyrics = generate_lyrics(prompt)

            # Similarity (SAFE)
            if lyrics and lyrics.strip():
                similarity_score = compute_similarity(
                    caption, lyrics, sbert
                )

        # ---------------- OUTPUT ----------------
        # Caption
        st.markdown(
        f"""
        <div class="caption-box">
        <div class="box-title">📝 Caption</div>
        {caption}
        </div>
        """,
        unsafe_allow_html=True
        )

        # Emotion
        st.markdown(
        f"""
        <div class="emotion-box">
        <div class="box-title">😊 Emotion</div>
        {emotion}
        </div>
        """,
        unsafe_allow_html=True
        )

        # Lyrics
        st.markdown(
        f"""
        <div class="lyrics-box">
        <div class="box-title">🎼 Lyrics</div>
        {lyrics}
        </div>
        """,
        unsafe_allow_html=True
        )

        # Similarity
        if similarity_score:
            st.markdown(
            f"""
            <div class="similarity-box">
            🔗 Caption–Lyrics Similarity: {similarity_score:.3f}
            </div>
            """,
            unsafe_allow_html=True

            )




