import streamlit as st
import os
import re
import torch
import numpy as np
import pronouncing
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
# GROQ CONFIG (Aligned with Paper: LLaMA 3.1)
# -------------------------------------------------
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
client = Groq()

def generate_lyrics(prompt):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",  # 120B parameter flagship open-weights model
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional songwriter who writes structured lyrics with consistent syllable rhythm and rhyme schemes."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,  # Higher limit so internal reasoning tokens don't cut off the lyrics
        )
        return response.choices[0].message.content

    except APIStatusError as e:
        st.error(f"Groq API error (HTTP {e.status_code}): {e.message}")
        return None
    except APIConnectionError:
        st.error("Could not connect to Groq. Please try again.")
        return None

# -------------------------------------------------
# UPGRADE 1: TOP-2 EMOTION BLENDING
# -------------------------------------------------
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

def detect_top2_emotions(text):
    model = load_emotion_model()
    outputs = model(text)[0]
    outputs_sorted = sorted(outputs, key=lambda x: x["score"], reverse=True)
    
    # Grab top 2 emotions and their confidence scores
    primary = outputs_sorted[0]
    secondary = outputs_sorted[1]
    
    emotion_summary = f"{primary['label'].capitalize()} ({primary['score']*100:.1f}%) + {secondary['label'].capitalize()} ({secondary['score']*100:.1f}%)"
    return emotion_summary, primary['label'], secondary['label']

# -------------------------------------------------
# BLIP IMAGE CAPTION (YOUR FINE-TUNED MODEL)
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
# SBERT SIMILARITY + UPGRADE 2: RHYTHM & RHYME METRICS
# -------------------------------------------------
@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity(caption, lyrics, sbert):
    embeddings = sbert.encode([caption, lyrics])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(score)

def count_word_syllables(word):
    clean_w = re.sub(r'[^a-zA-Z]', '', word.lower())
    if not clean_w:
        return 0
    phones = pronouncing.phones_for_word(clean_w)
    if phones:
        return pronouncing.syllable_count(phones[0])
    # Simple fallback vowel counter if word isn't in dictionary
    return max(1, len(re.findall(r'[aeiouy]+', clean_w)))

def evaluate_lyrics_structure(lyrics):
    # Get only actual lyric lines (ignore [Verse] / [Chorus] headers)
    lines = [
        line.strip() for line in lyrics.split("\n")
        if line.strip() and not line.strip().startswith("[")
    ]
    if len(lines) < 2:
        return 0.0, 0.0

    # 1. Count syllables per line
    syllable_counts = []
    last_words = []
    for line in lines:
        words = line.split()
        syllable_counts.append(sum(count_word_syllables(w) for w in words))
        last_word = re.sub(r'[^a-zA-Z]', '', words[-1].lower()) if words else ""
        last_words.append(last_word)

    avg_syllables = float(np.mean(syllable_counts))

    # 2. Check how many lines rhyme with another line in the song
    rhyming_lines = 0
    for i, word in enumerate(last_words):
        if not word:
            continue
        valid_rhymes = set(pronouncing.rhymes(word))
        other_words = last_words[:i] + last_words[i+1:]
        # Count if it rhymes (or shares the exact ending sound) with another line
        if any((other in valid_rhymes or (len(word) > 3 and other.endswith(word[-2:]))) for other in other_words if other):
            rhyming_lines += 1

    rhyme_density = rhyming_lines / len(lines)
    return avg_syllables, rhyme_density

# -------------------------------------------------
# UPGRADE 3: STRUCTURED SONG PROMPT BUILDER
# -------------------------------------------------
def build_prompt(caption, primary_emo, secondary_emo, optional_text=None):
    return f"""
You are an experienced songwriter.

Transform the visual scenario into structured, emotional song lyrics.

Scenario: {caption}
Primary Emotion: {primary_emo}
Secondary Undertone: {secondary_emo}
{f"Additional context: {optional_text}" if optional_text else ""}

Strict Structural & Musical Rules:
1. Write exactly 1 [Verse] (4 lines) and 1 [Chorus] (4 lines).
2. Follow a strict AABB or ABAB rhyme scheme at the end of each line.
3. Keep a steady rhythm of 8 to 10 syllables per line so it can be sung to a beat.
4. Blend the primary emotion ({primary_emo}) with subtle hints of {secondary_emo}.
5. Focus on poetic imagery and sensory details rather than literal descriptions.
6. Output ONLY the [Verse] and [Chorus] lyrics.
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

            # 1. Generate & Clean Caption
            caption = generate_caption(image, blip_p, blip_m)
            caption = caption.replace("< start >", "").replace("< end >", "").replace("start", "").replace("end", "").strip()

            # 2. Top-2 Emotion Detection
            combined_text = f"{caption} {optional_text}".strip()
            emotion_display, primary_emo, secondary_emo = detect_top2_emotions(combined_text)

            # 3. Generate Structured Lyrics (with 1 automatic retry if rhyme is too low)
            prompt = build_prompt(caption, primary_emo, secondary_emo, optional_text)
            lyrics = generate_lyrics(prompt)

            if lyrics and lyrics.strip():
                avg_syllables, rhyme_density = evaluate_lyrics_structure(lyrics)
                # Self-correction: if rhyme density is weak (< 0.4), regenerate once
                if rhyme_density < 0.4:
                    lyrics = generate_lyrics(prompt + "\nIMPORTANT: Make sure the last words of every pair of lines strictly rhyme!")
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
            <div class="box-title">😊 Affective Blend (Top-2 Emotions)</div>
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
                🔗 Semantic Similarity: {similarity_score:.3f} &nbsp;|&nbsp; 
                🥁 Avg Syllables/Line: {avg_syllables:.1f} &nbsp;|&nbsp; 
                🎵 Rhyme Score: {rhyme_density*100:.0f}%
                </div>
                """,
                unsafe_allow_html=True
            )
