# 🎵 Image-to-Lyrics Generator

An end-to-end multimodal AI app that takes an **image** and generates **structured song lyrics** that match what the image shows and how it feels.

It combines a fine-tuned image-captioning model, emotion classification, prompt engineering and an LLM into a single Streamlit web app.

---

## 📌 Overview

Most captioning systems stop at describing a picture. This project goes further: it works out **what is in the image** and **what emotion it carries**, then turns both into a verse and a chorus.

**Pipeline**

```
Uploaded Image (+ optional text context)
    │
    ▼
Fine-tuned BLIP (Flickr8K)            →  Image caption
    │
    ▼
DistilRoBERTa emotion classifier      →  Top-2 emotion blend
    │
    ▼
Prompt builder (caption + emotions)
    │
    ▼
LLM via Groq API (GPT-OSS-120B)       →  Verse + Chorus lyrics
    │
    ▼
Quality metrics                       →  Similarity, syllables, rhyme score
```

---

## ✨ Features

- **Fine-tuned BLIP** captioning model trained on **Flickr8K** (hosted on Hugging Face: `lokeshreddyayyaswamy/Blip_Flickr8k_Finetuned`)
- **Top-2 emotion blend** using `j-hartmann/emotion-english-distilroberta-base`, so the lyrics capture a primary emotion and a secondary undertone
- **Prompt engineering** that combines the caption, both emotions and optional user context, and asks for a rhymed verse and chorus (AABB or ABAB, about 8 to 10 syllables per line)
- **Optional extra context** box so the user can steer the mood or theme
- **Built-in quality metrics** shown with every result:
  - Caption–lyrics **cosine similarity** (Sentence-BERT, `all-MiniLM-L6-v2`)
  - **Average syllables per line**
  - **Rhyme score** (percentage of lines that share a phonetic ending)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| App | Streamlit |
| Image captioning | BLIP (fine-tuned), PyTorch, Hugging Face Transformers |
| Emotion detection | DistilRoBERTa (Hugging Face pipeline) |
| Lyrics generation | Groq API, GPT-OSS-120B |
| Evaluation | Sentence-Transformers, scikit-learn, BLEU, CIDEr, CLIP |
| Dataset | Flickr8K |

---

## 📂 Project Structure

```
Image_to_Lyrics/
├── .devcontainer/      # Dev container configuration
├── app.py              # Streamlit app (full pipeline)
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/lokeshreddyayyaswamy/Image_to_Lyrics.git
cd Image_to_Lyrics
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your Groq API key**

Create a file at `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

> Never commit this file. Keep `.streamlit/secrets.toml` in `.gitignore`.

**4. Run the app**

```bash
streamlit run app.py
```

Then upload a `.jpg` or `.png` image, optionally add extra context, and click **Generate Lyrics 🎶**.

---

## 📊 Evaluation

**In the app (for every generated song)**

| Metric | What it shows |
|---|---|
| Caption–lyrics similarity | How closely the lyrics stay on the theme of the image |
| Avg syllables per line | Whether the lines have a song-like rhythm |
| Rhyme score | How many lines share a rhyme ending |

**During model development (captioning and generation quality)**

| Metric | Score |
|---|---|
| BLEU | `<add score>` |
| CIDEr | `<add score>` |
| CLIP similarity | `<add score>` |
| Sentence-embedding cosine similarity | `<add score>` |

---

## 🖼️ Sample Output

| Input Image | Caption | Emotion Blend |
|---|---|---|
| `<add image>` | `<add caption>` | `<e.g. Joy (62.4%) + Surprise (20.1%)>` |

**Generated lyrics**

```
<paste one sample output here>
```

> Tip: add 1 or 2 screenshots of the app here. It's the first thing a recruiter will look at.

---

## 🔮 Future Improvements

- Deploy the app online (Streamlit Community Cloud or Hugging Face Spaces)
- Let users choose a music genre, language or lyric length
- Train on larger captioning datasets such as Flickr30K or COCO
- Add human evaluation of lyric quality and creativity

---

## 👤 Author

**A Lokesh**
B.Tech CSE (Artificial Intelligence), Madanapalle Institute of Technology and Science

[LinkedIn](https://www.linkedin.com/in/a-lokesh-a73272254) · [GitHub](https://github.com/lokeshreddyayyaswamy) · lokeshreddy2680@gmail.com
