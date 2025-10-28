import streamlit as st
from transformers import pipeline
import random
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Initialize sentiment analysis pipeline
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

# Customer service response templates
CUSTOMER_RESPONSES = {
    "POSITIVE": {
        "acknowledgment": [
            "I'm thrilled to hear that! 😊",
            "That's wonderful news! 🌟",
            "I'm so glad you're satisfied!",
            "Fantastic! Your happiness is our priority!"
        ],
        "engagement": [
            "How else can I help make your experience even better?",
            "Is there anything else I can assist you with today?",
            "What other services can I help you explore?",
            "I'd love to help you with anything else you need!"
        ]
    },
    "NEGATIVE": {
        "acknowledgment": [
            "I sincerely apologize for this experience. 😔",
            "I understand your frustration, and I'm here to help.",
            "I'm sorry to hear you're facing this issue.",
            "Thank you for bringing this to our attention."
        ],
        "resolution": [
            "Let me work on resolving this for you right away.",
            "I'll do everything I can to make this right.",
            "Can you provide more details so I can assist you better?",
            "I'm escalating this to ensure you get the best solution."
        ],
        "empathy": [
            "Your concerns are completely valid.",
            "I can understand why this is frustrating for you.",
            "You deserve better service, and I'm here to ensure that.",
            "I appreciate your patience as we work through this."
        ]
    },
    "NEUTRAL": {
        "professional": [
            "Thank you for reaching out.",
            "I'm here to assist you.",
            "I understand your inquiry.",
            "Let me help you with that."
        ],
        "clarification": [
            "Could you provide more details about your concern?",
            "What specific information are you looking for?",
            "I'd be happy to explain further.",
            "Let me get you the information you need."
        ]
    }
}

# Issue categories for routing
ISSUE_KEYWORDS = {
    "billing": ["bill", "charge", "payment", "invoice", "refund", "cost", "price"],
    "technical": ["not working", "error", "bug", "crash", "broken", "issue", "problem"],
    "account": ["login", "password", "access", "account", "sign in", "username"],
    "product": ["product", "item", "order", "delivery", "shipping", "quality"],
    "general": ["help", "question", "info", "information", "how to"]
}

def classify_sentiment(text, sentiment_result):
    """Enhanced sentiment classification with confidence thresholds"""
    label = sentiment_result['label']
    score = sentiment_result['score']
    
    # More nuanced classification
    if label == "POSITIVE":
        if score > 0.85:
            return "POSITIVE", score, "high"
        elif score > 0.65:
            return "POSITIVE", score, "medium"
        else:
            return "NEUTRAL", score, "low"
    elif label == "NEGATIVE":
        if score > 0.85:
            return "NEGATIVE", score, "high"
        elif score > 0.65:
            return "NEGATIVE", score, "medium"
        else:
            return "NEUTRAL", score, "low"
    else:
        return "NEUTRAL", score, "medium"

def detect_issue_category(text):
    """Detect the category of customer issue"""
    text_lower = text.lower()
    for category, keywords in ISSUE_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return "general"

def generate_customer_response(sentiment, confidence_level, issue_category, user_message):
    """Generate contextually appropriate customer service response"""
    response_parts = []
    
    if sentiment == "POSITIVE":
        response_parts.append(random.choice(CUSTOMER_RESPONSES["POSITIVE"]["acknowledgment"]))
        response_parts.append(random.choice(CUSTOMER_RESPONSES["POSITIVE"]["engagement"]))
        
    elif sentiment == "NEGATIVE":
        # More empathetic for high-confidence negative sentiment
        if confidence_level == "high":
            response_parts.append(random.choice(CUSTOMER_RESPONSES["NEGATIVE"]["acknowledgment"]))
            response_parts.append(random.choice(CUSTOMER_RESPONSES["NEGATIVE"]["empathy"]))
        response_parts.append(random.choice(CUSTOMER_RESPONSES["NEGATIVE"]["resolution"]))
        
        # Add category-specific help
        if issue_category == "billing":
            response_parts.append("I'm connecting you with our billing specialist who can review your account.")
        elif issue_category == "technical":
            response_parts.append("Our technical team will investigate this immediately.")
        elif issue_category == "account":
            response_parts.append("Let me help you regain access to your account securely.")
            
    else:  # NEUTRAL
        response_parts.append(random.choice(CUSTOMER_RESPONSES["NEUTRAL"]["professional"]))
        response_parts.append(random.choice(CUSTOMER_RESPONSES["NEUTRAL"]["clarification"]))
    
    return " ".join(response_parts)

def calculate_metrics(messages):
    """Calculate performance metrics"""
    if not messages:
        return None
    
    user_messages = [m for m in messages if m["role"] == "user"]
    
    sentiments = [m["sentiment"] for m in user_messages]
    confidences = [m["confidence"] for m in user_messages]
    
    metrics = {
        "total_interactions": len(user_messages),
        "positive_count": sentiments.count("POSITIVE"),
        "negative_count": sentiments.count("NEGATIVE"),
        "neutral_count": sentiments.count("NEUTRAL"),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "sentiment_shift": calculate_sentiment_shift(user_messages)
    }
    
    return metrics

def calculate_sentiment_shift(user_messages):
    """Calculate if sentiment improved over conversation"""
    if len(user_messages) < 2:
        return "N/A"
    
    first_sentiment = user_messages[0]["sentiment"]
    last_sentiment = user_messages[-1]["sentiment"]
    
    sentiment_score = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
    
    shift = sentiment_score[last_sentiment] - sentiment_score[first_sentiment]
    
    if shift > 0:
        return "Improved ↑"
    elif shift < 0:
        return "Declined ↓"
    else:
        return "Stable →"

def main():
    st.set_page_config(
        page_title="Customer Service AI Chatbot",
        page_icon="🎧",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 10px 0;
        }
        .sentiment-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin: 5px;
        }
        .positive { background-color: #4caf50; color: white; }
        .negative { background-color: #f44336; color: white; }
        .neutral { background-color: #2196f3; color: white; }
        .high-confidence { border: 3px solid gold; }
        .chat-message {
            padding: 15px;
            border-radius: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 20%;
        }
        .bot-message {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            margin-right: 20%;
        }
        .issue-category {
            background-color: #fff3cd;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 11px;
            display: inline-block;
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🎧 Customer Service AI Chatbot")
    st.markdown("*Emotion-aware support for enhanced customer satisfaction*")
    
    # Load model
    with st.spinner("Initializing AI assistant..."):
        sentiment_analyzer = load_sentiment_model()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "satisfaction_rating" not in st.session_state:
        st.session_state.satisfaction_rating = None
    
    # Layout: Main chat + Analytics sidebar
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.header("📊 Performance Analytics")
        
        metrics = calculate_metrics(st.session_state.messages)
        
        if metrics:
            # Key Metrics
            st.metric("Total Interactions", metrics["total_interactions"])
            st.metric("Sentiment Detection Accuracy", f"{metrics['avg_confidence']:.1%}")
            st.metric("Sentiment Trend", metrics["sentiment_shift"])
            
            # Sentiment Distribution
            st.subheader("Sentiment Distribution")
            sentiment_data = pd.DataFrame({
                "Sentiment": ["Positive", "Negative", "Neutral"],
                "Count": [metrics["positive_count"], metrics["negative_count"], metrics["neutral_count"]]
            })
            
            fig = px.pie(sentiment_data, values="Count", names="Sentiment",
                        color="Sentiment",
                        color_discrete_map={"Positive": "#4caf50", 
                                          "Negative": "#f44336", 
                                          "Neutral": "#2196f3"})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment Timeline
            if len(st.session_state.messages) > 1:
                st.subheader("Sentiment Timeline")
                user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
                timeline_df = pd.DataFrame({
                    "Message": range(1, len(user_msgs) + 1),
                    "Confidence": [m["confidence"] for m in user_msgs],
                    "Sentiment": [m["sentiment"] for m in user_msgs]
                })
                
                fig2 = px.line(timeline_df, x="Message", y="Confidence", 
                             color="Sentiment",
                             color_discrete_map={"POSITIVE": "#4caf50", 
                                               "NEGATIVE": "#f44336", 
                                               "NEUTRAL": "#2196f3"})
                fig2.update_layout(height=250)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Start a conversation to see analytics")
        
        # Customer Satisfaction Survey
        st.subheader("📝 Customer Satisfaction")
        if st.session_state.messages:
            rating = st.slider(
                "Rate your experience (1-5)",
                min_value=1,
                max_value=5,
                value=st.session_state.satisfaction_rating or 3,
                key="satisfaction_slider"
            )
            
            if st.button("Submit Rating"):
                st.session_state.satisfaction_rating = rating
                st.success(f"Thank you for rating us {rating}/5 stars!")
                
                if rating >= 4:
                    st.balloons()
        
        # Action Buttons
        st.markdown("---")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.satisfaction_rating = None
            st.rerun()
        
        if st.button("📥 Export Chat Log", use_container_width=True):
            if st.session_state.messages:
                chat_df = pd.DataFrame([
                    {
                        "Timestamp": m.get("timestamp", ""),
                        "Role": m["role"],
                        "Message": m["content"],
                        "Sentiment": m.get("sentiment", "N/A"),
                        "Confidence": m.get("confidence", 0)
                    }
                    for m in st.session_state.messages
                ])
                csv = chat_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "chat_log.csv",
                    "text/csv"
                )
    
    with col1:
        st.header("💬 Customer Support Chat")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>Customer:</strong><br>
                            {msg['content']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Sentiment analysis display
                    confidence_class = "high-confidence" if msg.get('confidence_level') == 'high' else ""
                    st.markdown(f"""
                        <div>
                            <span class="sentiment-badge {msg['sentiment'].lower()} {confidence_class}">
                                {msg['sentiment']} ({msg['confidence']:.1%})
                            </span>
                            <span class="issue-category">
                                Category: {msg.get('issue_category', 'general').upper()}
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-message bot-message">
                            <strong>Support Agent:</strong><br>
                            {msg['content']}
                        </div>
                    """, unsafe_allow_html=True)
        
        # Input area
        st.markdown("---")
        user_input = st.text_area(
            "Describe your issue or question:",
            key="user_input",
            placeholder="Example: I was charged twice for my last order...",
            height=100
        )
        
        col_a, col_b = st.columns([1, 4])
        with col_a:
            send_button = st.button("Send 📤", use_container_width=True)
        
        # Process input
        if send_button and user_input:
            # Analyze sentiment
            with st.spinner("Analyzing your message..."):
                result = sentiment_analyzer(user_input)[0]
                sentiment, confidence, confidence_level = classify_sentiment(user_input, result)
                issue_category = detect_issue_category(user_input)
            
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "sentiment": sentiment,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "issue_category": issue_category,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Generate bot response
            bot_response = generate_customer_response(
                sentiment, confidence_level, issue_category, user_input
            )
            
            # Add bot message
            st.session_state.messages.append({
                "role": "bot",
                "content": bot_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.rerun()
        
        # Quick response templates
        if not st.session_state.messages:
            st.markdown("### 🎯 Quick Start Examples:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("😊 Positive Feedback", use_container_width=True):
                    st.session_state.temp_input = "Your service is amazing! I love the new features."
                    st.rerun()
            
            with col2:
                if st.button("😔 Complaint", use_container_width=True):
                    st.session_state.temp_input = "I'm very disappointed. My order hasn't arrived and I was charged twice."
                    st.rerun()
            
            with col3:
                if st.button("😐 General Inquiry", use_container_width=True):
                    st.session_state.temp_input = "Can you help me understand how to update my payment method?"
                    st.rerun()
            
            if "temp_input" in st.session_state:
                user_input = st.session_state.temp_input
                del st.session_state.temp_input
                
                result = sentiment_analyzer(user_input)[0]
                sentiment, confidence, confidence_level = classify_sentiment(user_input, result)
                issue_category = detect_issue_category(user_input)
                
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "confidence_level": confidence_level,
                    "issue_category": issue_category,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                bot_response = generate_customer_response(
                    sentiment, confidence_level, issue_category, user_input
                )
                
                st.session_state.messages.append({
                    "role": "bot",
                    "content": bot_response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.rerun()

if __name__ == "__main__":
    main()