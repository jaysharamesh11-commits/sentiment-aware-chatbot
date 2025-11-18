import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.express as px
from collections import defaultdict

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'sentiment_stats' not in st.session_state:
    st.session_state.sentiment_stats = {
        'joy': 0, 'gratitude': 0, 'satisfaction': 0, 'admiration': 0, 
        'love': 0, 'relief': 0, 'anger': 0, 'frustration': 0, 
        'sadness': 0, 'disappointment': 0, 'fear': 0, 'confusion': 0, 
        'disgust': 0, 'neutral': 0
    }
if 'emotion_trend' not in st.session_state:
    st.session_state.emotion_trend = []

class AdvancedSentimentAnalyzer:
    def __init__(self):
        # Expanded sentiment lexicon with more nuanced emotions
        self.sentiment_words = {
            # Positive emotions
            'joy': {
                'happy', 'joy', 'delighted', 'ecstatic', 'thrilled', 'overjoyed',
                'excited', 'bliss', 'cheerful', 'jubilant', 'elated', 'gleeful'
            },
            'gratitude': {
                'thanks', 'thank you', 'grateful', 'appreciate', 'appreciation',
                'blessed', 'obliged', 'indebted', 'thankful'
            },
            'satisfaction': {
                'satisfied', 'pleased', 'content', 'fulfilled', 'happy with',
                'good enough', 'meets expectations', 'well done'
            },
            'admiration': {
                'amazing', 'awesome', 'brilliant', 'excellent', 'fantastic',
                'wonderful', 'perfect', 'outstanding', 'impressive', 'superb'
            },
            'love': {
                'love', 'adore', 'cherish', 'treasure', 'fond of', 'affection'
            },
            'relief': {
                'relieved', 'phew', 'thank goodness', 'finally', 'at last',
                'weight off', 'burden lifted'
            },
            
            # Negative emotions
            'anger': {
                'angry', 'mad', 'furious', 'enraged', 'outraged', 'irate',
                'livid', 'fuming', 'seething', 'infuriated'
            },
            'frustration': {
                'frustrated', 'annoyed', 'irritated', 'aggravated', 'bothered',
                'fed up', 'sick of', 'had enough', 'exasperated'
            },
            'sadness': {
                'sad', 'unhappy', 'depressed', 'miserable', 'heartbroken',
                'disheartened', 'down', 'blue', 'melancholy', 'gloomy'
            },
            'disappointment': {
                'disappointed', 'let down', 'disheartened', 'dissatisfied',
                'unfulfilled', 'displeased', 'regret', 'dismayed'
            },
            'fear': {
                'worried', 'anxious', 'nervous', 'scared', 'afraid', 'fearful',
                'terrified', 'panicked', 'concerned', 'apprehensive'
            },
            'confusion': {
                'confused', 'bewildered', 'perplexed', 'puzzled', 'baffled',
                'lost', 'don\'t understand', 'unclear', 'ambiguous'
            },
            'disgust': {
                'disgusting', 'gross', 'revolting', 'nasty', 'horrible',
                'awful', 'terrible', 'vile', 'repulsive'
            }
        }
        
        self.intensifiers = {
            'very', 'really', 'extremely', 'absolutely', 'completely',
            'totally', 'utterly', 'incredibly', 'exceptionally'
        }
        
        self.negations = {'not', "n't", 'no', 'never', 'none', 'nothing', 'nowhere'}
        
        self.emojis = {
            'joy': '😊', 'gratitude': '🙏', 'satisfaction': '👍', 'admiration': '🌟',
            'love': '❤️', 'relief': '😌', 'anger': '😠', 'frustration': '😤',
            'sadness': '😢', 'disappointment': '😞', 'fear': '😨', 'confusion': '😕',
            'disgust': '🤢', 'neutral': '😐'
        }

    def analyze_sentiment(self, text: str):
        text = text.lower().strip()
        words = re.findall(r'\b\w+\b', text)
        
        emotion_scores = defaultdict(float)
        negation_multiplier = 1
        intensity = 1.0
        
        for i, word in enumerate(words):
            # Check for negations
            if word in self.negations:
                negation_multiplier = -0.5  # Reduce impact rather than reverse
                continue
                
            # Check for intensifiers
            if word in self.intensifiers:
                intensity = 1.5
                continue
                
            # Score emotions
            for emotion, emotion_words in self.sentiment_words.items():
                if word in emotion_words:
                    emotion_scores[emotion] += 1 * negation_multiplier * intensity
                    negation_multiplier = 1
                    intensity = 1.0
        
        # Determine primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            confidence = min(abs(primary_emotion[1]) / 3, 1.0)
            
            # Classify as positive/negative/neutral
            if primary_emotion[0] in ['joy', 'gratitude', 'satisfaction', 'admiration', 'love', 'relief']:
                overall_sentiment = 'positive'
            else:
                overall_sentiment = 'negative'
                
            return overall_sentiment, primary_emotion[0], confidence, dict(emotion_scores)
        else:
            return 'neutral', 'neutral', 0.5, {}

class EnhancedSentimentChatbot:
    def __init__(self):
        self.analyzer = AdvancedSentimentAnalyzer()
        
        self.response_templates = {
            "joy": [
                "That's wonderful to hear! Your happiness brightens my day! 😊 What can I help you with?",
                "I'm thrilled to sense your joy! It's contagious! 🌟 How may I assist you?",
                "Your cheerful energy is amazing! Let me help make your experience even better! ✨"
            ],
            "gratitude": [
                "You're very welcome! I'm glad I could be of help 🙏 Is there anything else you need?",
                "Thank you for your kind words! It's my pleasure to assist you 💫 How can I help further?",
                "I appreciate your gratitude! It motivates me to help you even better 🌈"
            ],
            "satisfaction": [
                "I'm pleased to hear you're satisfied! That's what we aim for 👍 What else can I do for you?",
                "Great to know things are working well for you! 🎯 How can I continue to assist?",
                "Satisfaction guaranteed! I'm here to maintain that standard 💪"
            ],
            "admiration": [
                "Thank you for the amazing compliment! You're making me blush 🌟 How can I help?",
                "I'm honored by your admiration! Let me live up to those expectations 🚀",
                "Your kind words inspire me to do even better! ✨ What can I assist with?"
            ],
            "love": [
                "That's so heartwarming! I'm here to give you more reasons to love us ❤️",
                "Your affection means the world! Let me make this experience even more lovable 💖",
                "I'm feeling the love! Thank you for making this interaction special 💕"
            ],
            "relief": [
                "I'm glad that brought you relief! Let's keep the solutions coming 😌",
                "Phew! Happy to help ease your mind 💆‍♂️ What else can I clarify for you?",
                "Relief is what we aim for! Let me help with anything else that's on your mind 🌈"
            ],
            "anger": [
                "I understand you're angry, and I want to help resolve this immediately 😔",
                "Your frustration is completely valid. Let me work to fix this for you 🛠️",
                "I hear the anger in your voice, and I'm committed to making this right 💪"
            ],
            "frustration": [
                "I can feel your frustration, and I'm here to smooth things out 🫂",
                "Let's tackle this frustration together - I'm on your side 🤝",
                "I understand how frustrating this must be. Let me find a solution right away 🔍"
            ],
            "sadness": [
                "I'm sorry to hear you're feeling down. Let me help brighten your day 🌞",
                "Your sadness matters to me. Let's work together to improve this situation 💝",
                "I'm here to help lift your spirits. What can I do to make things better? 🌈"
            ],
            "disappointment": [
                "I understand your disappointment, and I want to exceed your expectations 🎯",
                "Let me help turn this disappointment into satisfaction 🔄",
                "I hear your disappointment loud and clear. Let me make this right for you ✨"
            ],
            "fear": [
                "I understand your concerns. Let me help provide clarity and reassurance 🛡️",
                "There's no need to worry - I'm here to guide you through this 🗺️",
                "Let's address your fears together. You're in safe hands 🤗"
            ],
            "confusion": [
                "I understand this can be confusing. Let me clarify things for you 📚",
                "No worries about the confusion - I'm here to make everything clear 💡",
                "Let me help clear up any confusion. What specific part can I explain? 🔍"
            ],
            "disgust": [
                "I apologize for this unpleasant experience. Let me help fix this immediately 🧹",
                "I understand why you'd feel this way. Let me work to resolve this 🛠️",
                "Your feedback is important. Let me help improve this situation right away 💫"
            ],
            "neutral": [
                "Hello! How can I assist you today? 👋",
                "Hi there! What can I help you with? 🤔",
                "I'm here and ready to help! What do you need? 💼"
            ]
        }
    
    def get_response(self, user_input: str):
        overall_sentiment, primary_emotion, confidence, emotion_scores = self.analyzer.analyze_sentiment(user_input)
        
        # Get appropriate response template
        templates = self.response_templates.get(primary_emotion, self.response_templates["neutral"])
        response = np.random.choice(templates)
        
        # Add emotional emoji
        emoji = self.analyzer.emojis.get(primary_emotion, '😐')
        response = f"{emoji} {response}"
        
        return response, overall_sentiment, primary_emotion, confidence, emotion_scores

# Initialize chatbot
chatbot = EnhancedSentimentChatbot()

# Streamlit UI
st.set_page_config(page_title="Advanced Sentiment Chatbot", page_icon="🧠", layout="wide")

st.title("🧠 Advanced Emotion-Aware Chatbot")
st.markdown("Chat with me! I can understand complex emotions and respond with empathy.")

# Sidebar for advanced analytics
with st.sidebar:
    st.header("📊 Emotional Analytics")
    
    if st.session_state.conversation:
        # Emotion distribution
        emotion_counts = st.session_state.sentiment_stats.copy()
        # Remove emotions with zero count for cleaner chart
        emotion_counts = {k: v for k, v in emotion_counts.items() if v > 0}
        
        if emotion_counts:
            emotion_data = pd.DataFrame({
                'Emotion': list(emotion_counts.keys()),
                'Count': list(emotion_counts.values())
            })
            
            fig_pie = px.pie(emotion_data, values='Count', names='Emotion', 
                           title='Emotional Distribution',
                           color='Emotion')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # Emotion trend over time
        if len(st.session_state.emotion_trend) > 1:
            trend_data = pd.DataFrame(st.session_state.emotion_trend)
            fig_line = px.line(trend_data, x='timestamp', y='sentiment_score', 
                             title='Emotional Trend Over Time',
                             markers=True)
            st.plotly_chart(fig_line, use_container_width=True)
    
    # Emotion guide
    with st.expander("🎭 Emotion Guide"):
        st.markdown("""
        **Positive Emotions:**
        - 😊 Joy: Happiness, excitement
        - 🙏 Gratitude: Thankfulness, appreciation  
        - 👍 Satisfaction: Contentment, fulfillment
        - 🌟 Admiration: Praise, amazement
        - ❤️ Love: Affection, adoration
        - 😌 Relief: Ease, comfort
        
        **Negative Emotions:**
        - 😠 Anger: Frustration, rage
        - 😤 Frustration: Annoyance, irritation
        - 😢 Sadness: Unhappiness, disappointment
        - 😞 Disappointment: Let down, dismay
        - 😨 Fear: Worry, anxiety
        - 😕 Confusion: Uncertainty, puzzlement
        - 🤢 Disgust: Revulsion, distaste
        """)

# Main chat area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Emotional Conversation")
    
    # Display conversation with emotional context
    for msg in st.session_state.conversation:
        if msg['type'] == 'user':
            with st.chat_message("user"):
                st.write(f"**You:** {msg['content']}")
                if 'primary_emotion' in msg:
                    emotion = msg['primary_emotion']
                    emoji = chatbot.analyzer.emojis.get(emotion, '😐')
                    confidence = msg.get('confidence', 0)
                    st.caption(f"Detected: {emotion.title()} {emoji} (Confidence: {confidence:.2f})")
                    
                    # Show emotion breakdown for high-confidence detections
                    if confidence > 0.7 and 'emotion_scores' in msg:
                        scores = msg['emotion_scores']
                        top_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                        if len(top_emotions) > 1:
                            emotion_text = " | ".join([f"{e[0]}: {e[1]:.1f}" for e in top_emotions])
                            st.caption(f"Emotion mix: {emotion_text}")
        else:
            st.chat_message("assistant").write(f"**Bot:** {msg['content']}")

with col2:
    st.subheader("🎯 Emotional Examples")
    
    tab1, tab2 = st.tabs(["Positive", "Negative"])
    
    with tab1:
        st.markdown("""
        **😊 Joy & Excitement:**
        - "I'm absolutely thrilled with this!"
        - "This makes me so happy!"
        - "I'm overjoyed with the results!"
        
        **🙏 Gratitude:**
        - "Thank you so much for your help!"
        - "I'm incredibly grateful for this"
        - "You've been a lifesaver!"
        
        **🌟 Admiration:**
        - "This is absolutely brilliant!"
        - "You guys are amazing!"
        - "Outstanding work!"
        """)
    
    with tab2:
        st.markdown("""
        **😠 Anger & Frustration:**
        - "I'm furious about this situation!"
        - "This is making me so angry!"
        - "I've had enough of this!"
        
        **😢 Sadness:**
        - "This makes me really sad"
        - "I'm feeling quite down about this"
        - "This is heartbreaking"
        
        **😨 Fear & Worry:**
        - "I'm really worried about this"
        - "This situation scares me"
        - "I'm anxious about what happens next"
        """)

# Emotional quick test buttons
st.subheader("🚀 Quick Emotional Test")
col1, col2, col3 = st.columns(3)

test_messages = {
    "Joy": "I'm absolutely thrilled and overjoyed with your amazing service! This is fantastic!",
    "Gratitude": "Thank you so much for your incredible help, I'm truly grateful for your support!",
    "Frustration": "I'm really frustrated and annoyed with this ongoing issue, it's exasperating!",
    "Worry": "I'm quite worried and anxious about this situation, it's really concerning me",
    "Confusion": "I'm completely confused and bewildered by these instructions, nothing makes sense",
    "Admiration": "This is absolutely brilliant and outstanding work, I'm truly impressed!"
}

for i, (emotion, message) in enumerate(test_messages.items()):
    col = [col1, col2, col3][i % 3]
    with col:
        if st.button(f"Test {emotion}", key=f"test_{emotion}"):
            # Simulate user input
            st.session_state.conversation.append({
                'type': 'user', 
                'content': message,
                'timestamp': datetime.now()
            })
            
            # Get bot response
            response, overall_sentiment, primary_emotion, confidence, emotion_scores = chatbot.get_response(message)
            
            # Add bot response
            st.session_state.conversation.append({
                'type': 'assistant',
                'content': response,
                'timestamp': datetime.now()
            })
            
            # Update user message with emotion info
            st.session_state.conversation[-2].update({
                'sentiment': overall_sentiment,
                'primary_emotion': primary_emotion,
                'confidence': confidence,
                'emotion_scores': emotion_scores
            })
            
            # Update statistics - SAFE ACCESS
            if primary_emotion in st.session_state.sentiment_stats:
                st.session_state.sentiment_stats[primary_emotion] += 1
            else:
                # Initialize if emotion doesn't exist (shouldn't happen with our fixed init)
                st.session_state.sentiment_stats[primary_emotion] = 1
            
            # Update emotion trend
            sentiment_score = 1 if overall_sentiment == 'positive' else -1 if overall_sentiment == 'negative' else 0
            st.session_state.emotion_trend.append({
                'timestamp': datetime.now(),
                'sentiment_score': sentiment_score,
                'emotion': primary_emotion
            })
            
            st.rerun()

# Chat input
st.markdown("---")
user_input = st.chat_input("Express your feelings or ask for help...")

if user_input:
    # Add user message to conversation
    user_msg = {
        'type': 'user', 
        'content': user_input,
        'timestamp': datetime.now()
    }
    st.session_state.conversation.append(user_msg)
    
    # Get bot response with emotional analysis
    response, overall_sentiment, primary_emotion, confidence, emotion_scores = chatbot.get_response(user_input)
    
    # Add bot response
    st.session_state.conversation.append({
        'type': 'assistant',
        'content': response,
        'timestamp': datetime.now()
    })
    
    # Update user message with detailed emotion info
    user_msg.update({
        'sentiment': overall_sentiment,
        'primary_emotion': primary_emotion,
        'confidence': confidence,
        'emotion_scores': emotion_scores
    })
    
    # Update statistics - SAFE ACCESS
    if primary_emotion in st.session_state.sentiment_stats:
        st.session_state.sentiment_stats[primary_emotion] += 1
    else:
        # Initialize if emotion doesn't exist
        st.session_state.sentiment_stats[primary_emotion] = 1
    
    # Update emotion trend
    sentiment_score = 1 if overall_sentiment == 'positive' else -1 if overall_sentiment == 'negative' else 0
    st.session_state.emotion_trend.append({
        'timestamp': datetime.now(),
        'sentiment_score': sentiment_score,
        'emotion': primary_emotion
    })
    
    st.rerun()

# Emotional intelligence metrics
if st.session_state.conversation:
    with st.expander("🧠 Emotional Intelligence Report"):
        if st.session_state.emotion_trend:
            recent_trend = st.session_state.emotion_trend[-10:]  # Last 10 interactions
            if recent_trend:
                avg_sentiment = np.mean([t['sentiment_score'] for t in recent_trend])
                st.metric("Current Emotional Climate", 
                         "Positive" if avg_sentiment > 0.3 else "Negative" if avg_sentiment < -0.3 else "Neutral",
                         f"{avg_sentiment:.2f}")
        
        total_interactions = len([m for m in st.session_state.conversation if m['type'] == 'user'])
        if total_interactions > 0:
            positive_pct = len([m for m in st.session_state.conversation 
                              if m.get('sentiment') == 'positive']) / total_interactions * 100
            st.metric("Positive Interaction Rate", f"{positive_pct:.1f}%")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Advanced Emotion-Aware Chatbot | Understanding the full spectrum of human emotions"
    "</div>", 
    unsafe_allow_html=True
)
