# Document Summary Assistant

A web-based document summarization application that allows users to upload PDF documents and generate concise or detailed summaries.

## Features

- Upload PDF documents
- Extract text from PDF files
- Generate document summaries
- Multiple summary-length options
- AI-powered summarization using OpenAI API
- Local fallback summarization when AI service is unavailable
- Simple and responsive web interface
- Error handling for invalid files and empty documents
- Flask REST API backend

## Tech Stack

- Python
- Flask
- HTML
- CSS
- JavaScript
- PyPDF
- OpenAI API
- python-dotenv

## Architecture

User
↓
Web Interface
↓
Flask Backend
↓
PDF Text Extraction
↓
Summarization Engine
↓
AI API / Local Fallback
↓
Summary Display

## How It Works

1. User uploads a PDF document.
2. Flask receives the uploaded file.
3. The PDF is processed using PyPDF.
4. Text is extracted from the document.
5. The user selects the desired summary length.
6. The backend generates the summary.
7. The generated summary is returned to the frontend.
8. The summary is displayed to the user.

## Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
cd document-summary-assistant