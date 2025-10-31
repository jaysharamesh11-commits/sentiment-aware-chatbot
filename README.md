💬 Sentiment-Aware, Multilingual & Dynamic Knowledge Chatbots

Combining emotion intelligence, language diversity, and adaptive learning through AI

🧠 Project 1: Sentiment-Aware Chatbot
📌 Description

This chatbot is designed to understand human emotions and respond empathetically.
By integrating a sentiment analysis model, it classifies messages as positive, negative, or neutral, then generates responses that align with the user’s mood.
The system uses a clean, Streamlit-based UI to ensure smooth and interactive conversations.

⚡ Key Features

Real-time emotion detection through text analysis

Context-aware and emotion-driven responses

Built on a modular architecture, allowing easy integration with future NLP models

Streamlit interface for dynamic interaction

🧩 Tech Stack

Python

Streamlit

Transformers (Hugging Face)

TextBlob / VADER Sentiment

📂 File

sample.py

🌍 Project 2: Multilingual Chatbot
📌 Description

The multilingual chatbot extends communication beyond English, offering real-time language detection, translation, and response generation across multiple languages.
It ensures inclusivity and cultural sensitivity by tailoring responses based on detected language.
Users can interact seamlessly without switching between languages manually.

⚡ Key Features

Automatic language detection using langdetect

Supports multiple languages (English, Hindi, Tamil, and more)

Bidirectional translation — user input and bot output

Contextually relevant and culturally aware responses

Interactive UI built with Streamlit

🧩 Tech Stack

Python

Streamlit

LangDetect

GoogleTrans / Deep Translator

Transformers

📂 File

multilingual_chatbot.py

💡 Project 3: Sentiment-Aware Chatbot with Dynamic Knowledge Expansion
📌 Description

This upgraded version fuses sentiment analysis, semantic understanding, and real-time knowledge expansion into one intelligent chatbot.
Using ChromaDB as a vector database, the chatbot continuously learns from new information sources — allowing it to update its knowledge base dynamically and respond with current, context-rich insights.
The system blends emotional intelligence with adaptive reasoning for a more human-like conversation experience.

⚡ Key Features

🧠 Dynamic Knowledge Integration — Expands its internal database with new data periodically

💬 Emotion + Context Fusion — Understands tone and retrieves emotionally relevant information

🔍 Semantic Search — Finds conceptually related answers beyond exact keywords

🖥️ Aesthetic UI — Pink background with black text for a friendly and modern interface

⚙️ Periodic Updates — Can auto-refresh knowledge base from web or local data sources

⚡ Streamlit-powered interaction with persistent sessions

🧩 Tech Stack

Python

Streamlit

ChromaDB

Tenacity

Transformers

📂 File

dynamic_knowledge.py

⚙️ Installation

Clone the repository:

git clone https://github.com/jaysharamesh11-commits/sentiment-aware-chatbot.git
cd sentiment-aware-chatbot


Install dependencies:

pip install -r requirements.txt

🚀 How to Run
🧠 Sentiment Chatbot
streamlit run sample.py

🌍 Multilingual Chatbot
streamlit run multilingual_chatbot.py

💡 Dynamic Knowledge Chatbot
streamlit run dynamic_knowledge.py

🖼️ Screenshots

Screenshots for each chatbot are available in the Screenshots/ directory:

Sentiment Chatbot → Screenshots/sentiment_ui.png

Multilingual Chatbot → Screenshots/multilingual_ui.png

Dynamic Chatbot → Screenshots/dynamic_ui.png

🧰 Common Dependencies

Python 3.10+

Streamlit

Transformers

LangDetect

Deep Translator

ChromaDB

Tenacity

👩‍💻 Author

Jaysha Ramesh
College Student | Data Science