import streamlit as st
from transformers import pipeline
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
import time

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Language configuration
SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'flag': '🇬🇧'},
    'es': {'name': 'Spanish', 'flag': '🇪🇸'},
    'hi': {'name': 'Hindi', 'flag': '🇮🇳'},
    'fr': {'name': 'French', 'flag': '🇫🇷'}
}

# Cache the sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    """Load multilingual sentiment analysis model"""
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )

def detect_language(text):
    """Detect the language of input text"""
    try:
        lang = detect(text)
        # Map detected language to supported languages
        if lang in SUPPORTED_LANGUAGES:
            return lang
        return 'en'  # Default to English
    except LangDetectException:
        return 'en'

def translate_text(text, source_lang, target_lang):
    """Translate text between languages"""
    if source_lang == target_lang:
        return text
    
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        return translator.translate(text)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def analyze_sentiment(text):
    """Analyze sentiment using multilingual model"""
    sentiment_analyzer = load_sentiment_model()
    result = sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens
    
    # Map labels to emotions
    label_map = {
        'positive': 'Positive',
        'negative': 'Negative',
        'neutral': 'Neutral'
    }
    
    sentiment = label_map.get(result['label'].lower(), result['label'])
    confidence = result['score']
    
    return sentiment, confidence

def generate_response(sentiment, user_input, detected_lang):
    """Generate culturally appropriate response based on sentiment and language"""
    
    # Response templates by language and sentiment
    responses = {
        'en': {
            'Positive': [
                f"That's wonderful! I'm so glad to hear you're feeling positive. 😊",
                f"Your positive energy is contagious! Keep that spirit up!",
                f"It's great to sense such enthusiasm in your words!"
            ],
            'Negative': [
                f"I'm sorry you're feeling this way. I'm here to listen. 💙",
                f"That sounds challenging. Would you like to talk more about it?",
                f"I understand this is difficult. Take your time, I'm here for you."
            ],
            'Neutral': [
                f"I see. Tell me more about what's on your mind.",
                f"I'm listening. How can I help you today?",
                f"Thanks for sharing. What would you like to discuss?"
            ]
        },
        'es': {
            'Positive': [
                f"¡Qué maravilloso! Me alegra mucho saber que te sientes positivo. 😊",
                f"¡Tu energía positiva es contagiosa! ¡Mantén ese espíritu!",
                f"¡Es genial sentir tanto entusiasmo en tus palabras!"
            ],
            'Negative': [
                f"Lamento que te sientas así. Estoy aquí para escucharte. 💙",
                f"Eso suena desafiante. ¿Te gustaría hablar más al respecto?",
                f"Entiendo que esto es difícil. Tómate tu tiempo, estoy aquí para ti."
            ],
            'Neutral': [
                f"Ya veo. Cuéntame más sobre lo que piensas.",
                f"Te escucho. ¿Cómo puedo ayudarte hoy?",
                f"Gracias por compartir. ¿Qué te gustaría discutir?"
            ]
        },
        'hi': {
            'Positive': [
                f"यह बहुत अच्छा है! मुझे खुशी है कि आप सकारात्मक महसूस कर रहे हैं। 😊",
                f"आपकी सकारात्मक ऊर्जा संक्रामक है! इस भावना को बनाए रखें!",
                f"आपके शब्दों में इतना उत्साह देखना बहुत अच्छा है!"
            ],
            'Negative': [
                f"मुझे खेद है कि आप ऐसा महसूस कर रहे हैं। मैं सुनने के लिए यहाँ हूँ। 💙",
                f"यह चुनौतीपूर्ण लगता है। क्या आप इसके बारे में और बात करना चाहेंगे?",
                f"मैं समझता हूँ कि यह कठिन है। अपना समय लें, मैं आपके लिए यहाँ हूँ।"
            ],
            'Neutral': [
                f"मैं समझा। मुझे अपने मन की बात और बताएं।",
                f"मैं सुन रहा हूँ। आज मैं आपकी कैसे मदद कर सकता हूँ?",
                f"साझा करने के लिए धन्यवाद। आप क्या चर्चा करना चाहेंगे?"
            ]
        },
        'fr': {
            'Positive': [
                f"C'est merveilleux! Je suis ravi d'apprendre que vous vous sentez positif. 😊",
                f"Votre énergie positive est contagieuse! Gardez cet esprit!",
                f"C'est formidable de ressentir autant d'enthousiasme dans vos mots!"
            ],
            'Negative': [
                f"Je suis désolé que vous vous sentiez ainsi. Je suis là pour écouter. 💙",
                f"Cela semble difficile. Aimeriez-vous en parler davantage?",
                f"Je comprends que c'est difficile. Prenez votre temps, je suis là pour vous."
            ],
            'Neutral': [
                f"Je vois. Parlez-moi davantage de ce qui vous préoccupe.",
                f"Je vous écoute. Comment puis-je vous aider aujourd'hui?",
                f"Merci de partager. De quoi aimeriez-vous discuter?"
            ]
        }
    }
    
    # Get response in detected language
    lang_responses = responses.get(detected_lang, responses['en'])
    sentiment_responses = lang_responses.get(sentiment, lang_responses['Neutral'])
    
    import random
    return random.choice(sentiment_responses)

def main():
    st.set_page_config(
        page_title="Multilingual Sentiment-Aware Chatbot",
        page_icon="🌍",
        layout="wide"
    )
    
    # Header
    st.title("🌍 Multilingual Sentiment-Aware Chatbot")
    st.markdown("### Powered by AI • Supports English, Spanish, Hindi & French")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot features:
        - 🌐 **Auto Language Detection**
        - 🎭 **Multilingual Sentiment Analysis**
        - 💬 **Culturally Aware Responses**
        - 🔄 **Real-time Translation**
        
        **Supported Languages:**
        """)
        
        for code, info in SUPPORTED_LANGUAGES.items():
            st.markdown(f"{info['flag']} {info['name']}")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`")
    
    # Main chat interface
    st.markdown("---")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "metadata" in message:
                    with st.expander("🔍 Analysis Details"):
                        metadata = message["metadata"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Detected Language", 
                                     f"{SUPPORTED_LANGUAGES.get(metadata['lang'], {'flag': '🌐'})['flag']} {metadata['lang'].upper()}")
                        with col2:
                            st.metric("Sentiment", metadata['sentiment'])
                        with col3:
                            st.metric("Confidence", f"{metadata['confidence']:.2%}")
    
    # Chat input
    user_input = st.chat_input("Type your message in any supported language...")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Process message
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your message..."):
                # Detect language
                detected_lang = detect_language(user_input)
                
                # Translate to English if needed
                text_for_analysis = translate_text(user_input, detected_lang, 'en')
                
                # Analyze sentiment
                sentiment, confidence = analyze_sentiment(text_for_analysis)
                
                # Generate response
                response = generate_response(sentiment, user_input, detected_lang)
                
                # Display response
                st.markdown(response)
                
                # Show analysis details
                with st.expander("🔍 Analysis Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Detected Language", 
                                 f"{SUPPORTED_LANGUAGES.get(detected_lang, {'flag': '🌐'})['flag']} {detected_lang.upper()}")
                    with col2:
                        st.metric("Sentiment", sentiment)
                    with col3:
                        st.metric("Confidence", f"{confidence:.2%}")
        
        # Save assistant message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "metadata": {
                "lang": detected_lang,
                "sentiment": sentiment,
                "confidence": confidence
            }
        })
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit, Transformers & Deep-Translator"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()