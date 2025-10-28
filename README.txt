Sentiment-Aware Chatbot

This project implements an intelligent chatbot capable of detecting and adapting to user emotions in real time. It uses natural language processing (NLP) and machine learning models to identify user sentiment and tailor responses accordingly. The application is built using Streamlit for the frontend interface and Hugging Face Transformers for sentiment analysis.

Overview

The Sentiment-Aware Chatbot analyzes user input to determine emotional tone (positive, negative, or neutral) and generates responses that reflect empathy and awareness. This creates a more engaging and human-like conversation experience.

Features

Real-time chat interface built with Streamlit.

Automatic emotion detection using pre-trained NLP models.

Dynamic responses adjusted based on detected sentiment.

Visualization of sentiment trends over time.

Lightweight and easy to deploy on Streamlit Cloud.

Tech Stack
Component	Technology
Frontend	Streamlit
Backend	Python
NLP Model	Hugging Face Transformers (distilbert-base-uncased-finetuned-sst-2-english)
Visualization	Plotly
Deployment	Streamlit Cloud
Installation

Clone the repository

git clone https://github.com/jaysharamesh11-commits/sentiment-aware-chatbot
cd sentiment-aware-chatbot


Install dependencies

pip install -r requirements.txt


Run the application

streamlit run analysis.py

Project Structure
sentiment-aware-chatbot/
│
├── analysis.py             # Main application file
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
└── Screenshots/            # Example outputs (optional)

Deployment

To deploy on Streamlit Cloud:

Push your repository to GitHub.

Go to Streamlit Cloud
.

Connect your GitHub account and select the repository.

Choose main as the branch and analysis.py as the entry file.

Deploy the app.

Example Output
Sentiment	Probability
Positive	0.84
Neutral	0.10
Negative	0.06
Author

Jaysha Ramesh
Data Science Student 
GitHub: jaysharamesh11-commits

Future Enhancements

Implement memory-based responses for contextual awareness.

Add multi-language sentiment detection.

Integrate voice-based input and output.

Improve model accuracy with fine-tuning.
