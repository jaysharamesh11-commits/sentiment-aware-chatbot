Sentiment-Aware and Multilingual Chatbot
📌 Overview

This repository contains two independent yet related chatbot implementations developed as part of an internship project.
Each chatbot focuses on a different core capability — one analyzes sentiment, while the other handles multilingual interactions.

🧠 Project 1: Sentiment-Aware Chatbot

Description:
A chatbot that can understand user emotions through text input and respond empathetically.
It integrates sentiment analysis models to classify messages as positive, negative, or neutral, then tailors responses accordingly.

Key Features:

Real-time emotion detection

Adaptive and context-aware responses

Streamlit-based interface for user interaction

Expandable architecture for future emotion-driven applications

File: sample.py

🌍 Project 2: Multilingual Chatbot

Description:
An upgraded chatbot capable of automatic language detection, translation, and response generation across multiple languages.
This bot enhances accessibility and inclusivity for users worldwide.

Key Features:

Detects the user’s language automatically

Supports at least three additional languages beyond English

Provides culturally appropriate and context-sensitive responses

Uses langdetect, googletrans, and deep-translator for smooth multilingual support

File: multilingual_chatbot.py

⚙️ Installation

Clone the repository:

git clone https://github.com/jaysharamesh11-commits/sentiment-aware-chatbot.git
cd sentiment-aware-chatbot


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run sample.py


(or run multilingual_chatbot.py for the upgraded version)

🧩 Tech Stack

Python

Transformers

LangDetect

Deep Translator

GoogleTrans

Streamlit

📸 Screenshots

Include screenshots of both chatbot interfaces in the Screenshots/ directory.

📬 Author

Jaysha Ramesh
College Student | Data Science |