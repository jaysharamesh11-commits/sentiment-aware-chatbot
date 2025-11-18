Sentiment-Aware Customer Service Chatbot
Overview

This project implements a sentiment-aware customer support chatbot using Streamlit and HuggingFace Transformers.
The chatbot analyzes user messages, detects emotional tone, identifies issue categories, and generates appropriate customer service responses. It also includes a full analytics dashboard for visualizing sentiment trends and interaction quality.

Key Features
1. Real-Time Sentiment Analysis

Classifies messages as Positive, Negative, or Neutral.

Uses confidence scores to refine overall interpretation.

Adjusts tone of response based on sentiment intensity.

2. Issue Category Detection

Automatically categorizes user concerns into:

Billing

Technical

Account

Product

General inquiry

3. Intelligent Response Generation

Empathetic, context-aware replies

Category-specific support statements

Professional communication patterns

4. Performance Analytics Dashboard

Includes:

Sentiment distribution visualization

Confidence timeline

Interaction metrics (counts, trend, accuracy)

5. Chat Export

Full conversation exportable as CSV

6. Streamlit-Based Interface

Clean chat UI

Conversation controls

Quick-start example buttons

Satisfaction rating widget

File

analysis.py
Contains the full implementation of the chatbot, analytics system, and UI.

Installation
1. Clone the Repository
git clone https://github.com/jaysharamesh11-commits/sentiment-aware-chatbot.git
cd sentiment-aware-chatbot

2. Install Dependencies
pip install -r requirements.txt

Running the Application
streamlit run analysis.py

Requirements
streamlit
transformers
pandas
plotly
torch
numpy

Example User Inputs

Here are realistic examples used to demonstrate the chatbot’s behavior:

Positive Inputs

"Your customer service team was extremely helpful today."

"The new update works perfectly. Great job!"

"I love using your app. It makes everything easier."

Negative Inputs

"I was charged twice for the same order."

"My account keeps logging me out. This is really frustrating."

"The product arrived damaged and I want a refund."

These examples help demonstrate how the chatbot responds differently depending on sentiment and issue type.

Author

Jaysha Ramesh
Data Science Student
