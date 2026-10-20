import streamlit as st
import os
import re
import torch
import numpy as np
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline,
)
from groq import Groq, APIStatusError, APIConnectionError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Image to Lyrics", layout="centered")
st.title("🎵 Image → Lyrics Generator")
st.markdown("""
<style>
.caption-box{
    background:#1e3a8a;
    padding:16px;
    border-radius:10px;
    margin-bottom:18px;
    color:white;
}
.emotion-box{
    background:#6b21a8;
    padding:16px;
    border-radius:10px;
    margin-bottom:18px;
    color:white;
}
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
.similarity-box{
    background:#064e3b;
    padding:16px;
    border-radius:10px;
    color:#bbf7d0;
    font-weight:600;
}
.box-title{
    font-size:20px;
    font-weight:600;
    margin-bottom:6px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# GROQ CONFIG (120B Model)
# -------------------------------------------------
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
client = Groq()

def generate_lyrics(prompt):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional songwriter who writes structured lyrics with consistent syllable rhythm and rhyme schemes."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    except APIStatusError as e:
        st.error(f"Groq API error (HTTP {e.status_code}).")
        return None
    except APIConnectionError:
        st.error("Could not connect to Groq. Please try again.")
        return None

# -------------------------------------------------
# FIXED EMOTION MODEL (Handles all HF versions safely)
# -------------------------------------------------
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

def detect_top2_emotions(text):
    model = load_emotion_model()
    raw_outputs = model(text, top_k=None)

    # Safely unwrap whether HuggingFace returns [[dict, dict]] or [dict, dict]
    if isinstance(raw_outputs, list) and len(raw_outputs) > 0 and isinstance(raw_outputs[0], list):
        outputs = raw_outputs[0]
    else:
        outputs = raw_outputs

    outputs_sorted = sorted(outputs, key=lambda x: x["score"], reverse=True)

    primary = outputs_sorted[0]
    secondary = outputs_sorted[1] if len(outputs_sorted) > 1 else outputs_sorted[0]

    emotion_summary = (
        f"{primary['label'].capitalize()} ({primary['score']*100:.1f}%) + "
        f"{secondary['label'].capitalize()} ({secondary['score']*100:.1f}%)"
    )
    return emotion_summary, primary["label"], secondary["label"]

# -------------------------------------------------
# BLIP IMAGE CAPTION (OFFLINE / FINE-TUNED)
# -------------------------------------------------
@st.cache_resource
def load_blip(model_name):
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).eval()
    return processor, model

def generate_caption(image, processor, model):
    inputs = processor(image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=50,
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1.0
        )
    return processor.decode(output[0], skip_special_tokens=True)

# -------------------------------------------------
# SBERT SIMILARITY + BUILT-IN RHYTHM/RHYME METRICS
# -------------------------------------------------
@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity(caption, lyrics, sbert):
    embeddings = sbert.encode([caption, lyrics])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(score)

def count_word_syllables(word):
    word = re.sub(r'[^a-z]', '', word.lower())
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    # Remove silent trailing 'e', 'es', 'ed'
    word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
    word = re.sub(r'^y', '', word)
    vowels = re.findall(r'[aeiouy]{1,2}', word)
    return max(1, len(vowels))

def get_phonetic_ending(word):
    word = re.sub(r'[^a-z]', '', word.lower())
    if not word:
        return ""
    # Normalize common silent endings so rhyming words match cleanly
    word = re.sub(r'e$', '', word)
    match = re.search(r'[aeiouy]+[bcdfghjklmnpqrstvwxz]*$', word)
    return match.group(0) if match else word[-2:]

def evaluate_lyrics_structure(lyrics):
    lines = [
        line.strip() for line in lyrics.split("\n")
        if line.strip() and not line.strip().startswith("[") and not line.strip().startswith("(")
    ]
    if len(lines) < 2:
        return 0.0, 0.0

    syllable_counts = []
    endings = []
    for line in lines:
        words = [re.sub(r'[^a-zA-Z]', '', w) for w in line.split()]
        words = [w for w in words if w]
        if not words:
            continue
        syllable_counts.append(sum(count_word_syllables(w) for w in words))
        endings.append(get_phonetic_ending(words[-1]))

    if not syllable_counts:
        return 0.0, 0.0

    avg_syllables = float(np.mean(syllable_counts))

    # Check how many lines share a phonetic rhyme ending with another line
    rhyming_lines = 0
    for i, end in enumerate(endings):
        if not end:
            continue
        other_endings = endings[:i] + endings[i+1:]
        if any(end == other or (len(end) >= 2 and len(other) >= 2 and end[-2:] == other[-2:]) for other in other_endings):
            rhyming_lines += 1

    rhyme_density = rhyming_lines / len(endings)
    return avg_syllables, rhyme_density

# -------------------------------------------------
# PROMPT BUILDER
# -------------------------------------------------
def build_prompt(caption, primary_emo, secondary_emo, optional_text=None):
    return f"""
You are an experienced songwriter.

Transform the scenario into structured, emotional song lyrics.

Scenario: {caption}
Dominant Emotion: {primary_emo}
Secondary Undertone: {secondary_emo}
{f"Additional emotional direction: {optional_text}" if optional_text else ""}

Output format:
Write 1 [Verse] (4 lines) and 1 [Chorus] (4 lines).

Requirements:
* Use an AABB or ABAB rhyme scheme at the end of the lines.
* Keep each line around 8 to 10 syllables so it flows like a real song.
* Blend the feelings of {primary_emo} and {secondary_emo} using poetic imagery.
* Write ONLY the lyrics with [Verse] and [Chorus] headers.
"""

# -------------------------------------------------
# UI
# -------------------------------------------------
optional_text = st.text_input("Additional Context", placeholder="Enter any additional context (optional)")

image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file:
    image = Image.open(image_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(image, width=280)

    if st.button("Generate Lyrics 🎶"):
        similarity_score = None
        avg_syllables, rhyme_density = 0.0, 0.0

        with st.spinner("Running full AI pipeline..."):
            MODEL_NAME = "lokeshreddyayyaswamy/Blip_Flickr8k_Finetuned"
            blip_p, blip_m = load_blip(MODEL_NAME)
            sbert = load_sbert()

            # 1. Caption
            caption = generate_caption(image, blip_p, blip_m)
            caption = re.sub(r'<\s*start\s*>|<\s*end\s*>|^start\s+|\s+end$', '', caption, flags=re.IGNORECASE).strip()

            # 2. Emotion (Safe Top-2 Blend)
            combined_text = f"{caption} {optional_text}".strip()
            emotion_display, primary_emo, secondary_emo = detect_top2_emotions(combined_text)

            # 3. Lyrics
            prompt = build_prompt(caption, primary_emo, secondary_emo, optional_text)
            lyrics = generate_lyrics(prompt)

            # 4. Metrics
            if lyrics and lyrics.strip():
                avg_syllables, rhyme_density = evaluate_lyrics_structure(lyrics)
                similarity_score = compute_similarity(caption, lyrics, sbert)

        # ---------------- OUTPUT ----------------
        st.markdown(
            f"""
            <div class="caption-box">
            <div class="box-title">📝 Fine-Tuned BLIP Caption</div>
            {caption}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="emotion-box">
            <div class="box-title">😊 Emotion Blend (Top-2)</div>
            {emotion_display}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="lyrics-box">
            <div class="box-title">🎼 Structured Lyrics</div>
            {lyrics}
            </div>
            """,
            unsafe_allow_html=True
        )

        if similarity_score is not None:
            st.markdown(
                f"""
                <div class="similarity-box">
                🔗 Caption–Lyrics Similarity: {similarity_score:.3f} &nbsp;|&nbsp;
                🥁 Avg Syllables/Line: {avg_syllables:.1f} &nbsp;|&nbsp;
                🎵 Rhyme Score: {rhyme_density*100:.0f}%
                </div>
                """,
                unsafe_allow_html=True
            )
