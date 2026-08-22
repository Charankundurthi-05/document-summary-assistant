import os
import re
from collections import Counter

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

api_key = os.getenv("OPENAI_API_KEY")

client = None

if api_key:
    client = OpenAI(api_key=api_key)

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


def split_into_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 30
    ]


def local_summary(text, summary_type="detailed"):
    """
    Local extractive summarizer.

    It does not generate new information.
    It selects important sentences directly from
    the uploaded document.
    """

    sentences = split_into_sentences(text)

    if not sentences:
        return text[:2000]

    stop_words = {
        "the", "a", "an", "and", "or", "but", "if", "then",
        "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "for", "with", "as", "by",
        "at", "from", "that", "this", "these", "those",
        "it", "its", "they", "their", "them", "we", "our",
        "you", "your", "he", "she", "his", "her", "which",
        "who", "what", "when", "where", "how", "can", "could",
        "would", "should", "will", "may", "might", "do",
        "does", "did", "not", "than", "also", "such"
    }

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    word_counts = Counter(
        word for word in words
        if word not in stop_words
    )

    if not word_counts:
        return "\n".join(sentences[:5])

    scored_sentences = []

    for index, sentence in enumerate(sentences):

        sentence_words = re.findall(
            r"\b[a-zA-Z]{3,}\b",
            sentence.lower()
        )

        if not sentence_words:
            continue

        score = sum(
            word_counts[word]
            for word in sentence_words
            if word in word_counts
        )

        score = score / len(sentence_words)

        # Give slightly higher priority to sentences
        # containing important structural keywords.
        important_terms = [
            "important",
            "conclusion",
            "result",
            "objective",
            "method",
            "finding",
            "key",
            "significant",
            "therefore",
            "finally",
            "summary"
        ]

        lower_sentence = sentence.lower()

        for term in important_terms:
            if term in lower_sentence:
                score += 1.5

        scored_sentences.append(
            (score, index, sentence)
        )

    scored_sentences.sort(
        key=lambda x: x[0],
        reverse=True
    )

    if summary_type == "short":
        number_of_sentences = min(
            5,
            len(sentences)
        )

    elif summary_type == "bullet":
        number_of_sentences = min(
            8,
            len(sentences)
        )

    else:
        number_of_sentences = min(
            12,
            len(sentences)
        )

    selected = scored_sentences[:number_of_sentences]

    # Restore original document order.
    selected.sort(
        key=lambda x: x[1]
    )

    selected_sentences = [
        item[2]
        for item in selected
    ]

    if summary_type == "bullet":

        return "\n".join(
            f"• {sentence}"
            for sentence in selected_sentences
        )

    return " ".join(selected_sentences)


def summarize_with_openai(text, summary_type="detailed"):

    if not client:
        raise RuntimeError(
            "OpenAI API key is not configured."
        )

    if summary_type == "short":

        instruction = """
Create a concise summary of the document.

Include:
- Main topic
- 5-7 most important points
- Important conclusions

Keep it short and easy to understand.
"""

    elif summary_type == "bullet":

        instruction = """
Summarize the document using clear bullet points.

Organize the answer into:
- Main topic
- Key points
- Important facts
- Conclusions

Do not add information that is not present in the document.
"""

    else:

        instruction = """
Create a detailed but easy-to-read summary of the document.

Use the following structure:

1. Overview
2. Key Points
3. Important Details
4. Conclusions

Preserve important names, numbers, dates, definitions and technical terms.
Do not invent information that is not present in the document.
"""

    prompt = f"""
You are a document summarization assistant.

{instruction}

DOCUMENT:
--------------------
{text}
--------------------
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt
    )

    return response.output_text


def summarize_document(text, summary_type="detailed"):

    if not text.strip():
        raise ValueError(
            "No readable text was found in the PDF."
        )

    # Try AI summarization first.
    if client:

        try:

            summary = summarize_with_openai(
                text,
                summary_type
            )

            return summary, "AI"

        except Exception as e:

            print(
                "OpenAI API unavailable. "
                "Using local summarizer."
            )

            print(
                "API error:",
                str(e)
            )

    # Local fallback.
    summary = local_summary(
        text,
        summary_type
    )

    return summary, "Local"


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/summarize", methods=["POST"])
def summarize():

    try:

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No file was uploaded."
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "Please select a PDF file."
            }), 400

        if not allowed_file(file.filename):

            return jsonify({
                "success": False,
                "error": "Only PDF files are supported."
            }), 400

        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        text = extract_text_from_pdf(
            filepath
        )

        if not text:

            return jsonify({
                "success": False,
                "error": (
                    "Could not extract readable "
                    "text from this PDF."
                )
            }), 400

        summary_type = request.form.get(
            "summary_type",
            "detailed"
        )

        summary, method = summarize_document(
            text,
            summary_type
        )

        return jsonify({

            "success": True,

            "filename": filename,

            "summary": summary,

            "method": method

        })

    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "status": "healthy",

        "service": "Document Summary Assistant",

        "ai_api_available": client is not None

    })


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )