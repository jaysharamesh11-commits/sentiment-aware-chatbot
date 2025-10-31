import streamlit as st
from transformers import pipeline
import random
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
import os
from pathlib import Path
import time
import uuid

# Initialize models
@st.cache_resource
def load_models():
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return sentiment_analyzer

# Knowledge Base Manager with ChromaDB
class DynamicKnowledgeBase:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name="knowledge_base")
        except:
            self.collection = self.client.create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Metadata storage
        self.metadata_path = Path(persist_directory) / "kb_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load metadata from disk"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "last_update": None,
                "total_updates": 0,
                "sources": []
            }
    
    def save_metadata(self):
        """Save metadata to disk"""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_documents(self, texts, sources):
        """Add new documents to the knowledge base"""
        if not texts:
            return
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Prepare metadata for each document
        metadatas = [
            {
                "source": source,
                "added_at": datetime.now().isoformat(),
                "text_preview": text[:100]
            }
            for text, source in zip(texts, sources)
        ]
        
        # Add to ChromaDB
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Update metadata
        self.metadata["last_update"] = datetime.now().isoformat()
        self.metadata["total_updates"] += 1
        
        for source in sources:
            if source not in self.metadata["sources"]:
                self.metadata["sources"].append(source)
        
        self.save_metadata()
    
    def search(self, query, k=3):
        """Search for relevant documents using semantic similarity"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, self.collection.count())
            )
            
            if not results['documents'][0]:
                return []
            
            formatted_results = []
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                # Convert distance to similarity score (cosine distance to similarity)
                similarity = 1 - (distance / 2)  # Normalize to 0-1 range
                
                formatted_results.append({
                    "text": doc,
                    "source": metadata.get("source", "unknown"),
                    "relevance_score": float(similarity),
                    "added_at": metadata.get("added_at", "unknown")
                })
            
            return formatted_results
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def get_stats(self):
        """Get knowledge base statistics"""
        try:
            doc_count = self.collection.count()
        except:
            doc_count = 0
        
        return {
            "total_documents": doc_count,
            "last_update": self.metadata.get("last_update"),
            "sources": self.metadata.get("sources", []),
            "total_updates": self.metadata.get("total_updates", 0)
        }
    
    def get_all_documents(self, limit=20):
        """Retrieve all documents for viewing"""
        try:
            results = self.collection.get(limit=limit)
            
            documents = []
            for doc, metadata in zip(results['documents'], results['metadatas']):
                documents.append({
                    "text": doc,
                    "source": metadata.get("source", "unknown"),
                    "added_at": metadata.get("added_at", "unknown")
                })
            
            return documents
        except:
            return []
    
    def clear(self):
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(name="knowledge_base")
            self.collection = self.client.create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
            self.metadata = {
                "last_update": None,
                "total_updates": 0,
                "sources": []
            }
            self.save_metadata()
        except Exception as e:
            st.error(f"Error clearing database: {e}")

# Data source loaders
class DataSourceLoader:
    @staticmethod
    def load_from_file(file_path):
        """Load knowledge from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks (paragraphs)
            chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
            sources = [f"file:{os.path.basename(file_path)}"] * len(chunks)
            return chunks, sources
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return [], []
    
    @staticmethod
    def load_from_text(text, source_name="manual_input"):
        """Load knowledge from raw text"""
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        sources = [source_name] * len(chunks)
        return chunks, sources
    
    @staticmethod
    def load_sample_knowledge():
        """Load sample product/company knowledge"""
        knowledge = [
            "Our company offers 24/7 customer support with average response time of 2 minutes.",
            "We have a 30-day money-back guarantee on all products with no questions asked.",
            "Free shipping is available on orders over $50 within the continental US.",
            "Premium members get 20% discount on all purchases and early access to new products.",
            "Our technical support team is available via chat, email, and phone.",
            "We recently launched a new mobile app with enhanced features and better performance.",
            "Product returns can be initiated through our website or by contacting support.",
            "We accept all major credit cards, PayPal, and cryptocurrency payments.",
            "Our sustainability initiative aims to reduce packaging waste by 50% by 2025.",
            "New AI-powered recommendation engine was launched last month to personalize user experience.",
            "Customer satisfaction rate is currently at 96% based on recent surveys.",
            "We offer live chat support in English, Spanish, French, and German languages.",
            "Our warehouse operates in 15 countries with local distribution centers.",
            "Express delivery option delivers products within 24-48 hours in major cities.",
            "We have a loyalty program where customers earn points for every purchase."
        ]
        sources = ["company_knowledge"] * len(knowledge)
        return knowledge, sources

# Sentiment-aware response generator with KB integration
def generate_hybrid_response(sentiment, confidence, query, kb_results, user_name="Customer"):
    """Generate response combining sentiment analysis and knowledge base"""
    response_parts = []
    
    # Sentiment-based greeting
    if sentiment == "POSITIVE":
        greetings = [
            f"I'm so glad to help you, {user_name}! 😊",
            f"Great to hear from you, {user_name}! 🌟",
            f"Happy to assist, {user_name}! ✨"
        ]
        response_parts.append(random.choice(greetings))
    elif sentiment == "NEGATIVE":
        greetings = [
            f"I understand your concern, {user_name}. Let me help you with that. 💙",
            f"I'm sorry you're experiencing this, {user_name}. Here's what I can do:",
            f"I hear you, {user_name}. Let me provide some information that might help:"
        ]
        response_parts.append(random.choice(greetings))
    else:
        greetings = [
            f"Thank you for your question, {user_name}.",
            f"I'm here to help, {user_name}.",
            f"Let me assist you with that, {user_name}."
        ]
        response_parts.append(random.choice(greetings))
    
    # Add knowledge base information if available
    if kb_results:
        response_parts.append("\n\n**Based on our latest information:**")
        relevant_count = 0
        for i, result in enumerate(kb_results[:3], 1):  # Top 3 results
            if result['relevance_score'] > 0.5:  # Relevance threshold
                response_parts.append(f"\n\n• {result['text']}")
                if result.get('source'):
                    response_parts.append(f"\n  _(Source: {result['source']})_")
                relevant_count += 1
        
        if relevant_count == 0:
            response_parts.append("\n\nI don't have highly relevant information about that specific query, but I'm here to help in any way I can.")
    else:
        response_parts.append("\n\nI don't have specific information about that in my current knowledge base, but I'm here to help in any way I can.")
    
    # Add empathetic closing based on sentiment
    if sentiment == "NEGATIVE" and confidence > 0.7:
        response_parts.append("\n\n**Is there anything else I can help clarify or resolve for you?**")
    else:
        response_parts.append("\n\n**Feel free to ask if you need more information!**")
    
    return " ".join(response_parts)

# Auto-update scheduler
def should_update_kb(last_update, update_interval_hours=24):
    """Check if knowledge base should be updated"""
    if not last_update:
        return True
    
    last_update_time = datetime.fromisoformat(last_update)
    time_elapsed = datetime.now() - last_update_time
    return time_elapsed > timedelta(hours=update_interval_hours)

def main():
    st.set_page_config(
        page_title="AI Chatbot with ChromaDB Knowledge Base",
        page_icon="🧠",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .chat-message {
            padding: 15px;
            border-radius: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 15%;
        }
        .bot-message {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            margin-right: 15%;
        }
        .kb-result {
            background-color: #ffe6f0;
            padding: 10px;
            border-radius: 8px;
            margin: 6px 0;
            border-left: 4px solid  #ff4b91;
            color: black;
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
        .kb-stats {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 10px;
            color: white;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🧠 AI Chatbot with ChromaDB Knowledge Base")
    st.markdown("*Combining emotional intelligence with semantic search and up-to-date information*")
    
    # Load models
    with st.spinner("Loading AI models..."):
        sentiment_analyzer = load_models()
    
    # Initialize knowledge base
    if 'kb' not in st.session_state:
        st.session_state.kb = DynamicKnowledgeBase()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'kb_initialized' not in st.session_state:
        # Load initial sample knowledge
        stats = st.session_state.kb.get_stats()
        if stats['total_documents'] == 0:
            texts, sources = DataSourceLoader.load_sample_knowledge()
            st.session_state.kb.add_documents(texts, sources)
        st.session_state.kb_initialized = True
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.header("📚 Knowledge Base Manager")
        
        # KB Stats
        stats = st.session_state.kb.get_stats()
        st.markdown(f"""
            <div class="kb-stats">
                <h4>📊 ChromaDB Stats</h4>
                <p><strong>Total Documents:</strong> {stats['total_documents']}</p>
                <p><strong>Total Updates:</strong> {stats['total_updates']}</p>
                <p><strong>Last Update:</strong> {stats['last_update'][:19] if stats['last_update'] else 'Never'}</p>
                <p><strong>Unique Sources:</strong> {len(stats['sources'])}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Update Options
        st.subheader("🔄 Update Knowledge Base")
        
        update_method = st.radio(
            "Update Method:",
            ["Manual Text Input", "Upload File", "Auto-Update Check", "Load Sample Data"]
        )
        
        if update_method == "Manual Text Input":
            new_knowledge = st.text_area(
                "Enter new knowledge (separate paragraphs with blank lines):",
                height=150,
                placeholder="Example:\n\nWe now offer same-day delivery in major cities.\n\nOur new chatbot handles 95% of queries automatically."
            )
            source_name = st.text_input("Source name:", value="manual_update")
            
            if st.button("Add to Knowledge Base", type="primary"):
                if new_knowledge:
                    texts, sources = DataSourceLoader.load_from_text(
                        new_knowledge, 
                        f"{source_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    )
                    st.session_state.kb.add_documents(texts, sources)
                    st.success(f"✅ Added {len(texts)} new documents to ChromaDB!")
                    time.sleep(0.5)
                    st.rerun()
        
        elif update_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload text file (.txt):",
                type=['txt']
            )
            
            if uploaded_file and st.button("Process File", type="primary"):
                # Save temporarily
                temp_path = Path("temp_upload.txt")
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                texts, sources = DataSourceLoader.load_from_file(temp_path)
                st.session_state.kb.add_documents(texts, sources)
                st.success(f"✅ Added {len(texts)} documents from file!")
                
                # Cleanup
                temp_path.unlink()
                time.sleep(0.5)
                st.rerun()
        
        elif update_method == "Auto-Update Check":
            st.info("📅 Auto-update checks for new information at configured intervals")
            
            update_interval = st.slider(
                "Update interval (hours):",
                min_value=1,
                max_value=168,
                value=24
            )
            
            if st.button("Check for Updates Now", type="primary"):
                if should_update_kb(stats['last_update'], update_interval):
                    with st.spinner("Fetching new information..."):
                        # Simulate fetching new data
                        time.sleep(1)
                        new_data = [
                            "Updated policy: We now accept cryptocurrency payments including Bitcoin and Ethereum.",
                            "New feature: Voice-based customer support launched this week with AI voice recognition.",
                            "System upgrade: Response time improved by 40% after infrastructure update.",
                            "Partnership announcement: We've partnered with major carriers for faster shipping.",
                            "Security update: Two-factor authentication is now available for all accounts."
                        ]
                        sources = [f"auto_update_{datetime.now().strftime('%Y%m%d')}"] * len(new_data)
                        
                        st.session_state.kb.add_documents(new_data, sources)
                        st.success(f"✅ Knowledge base updated with {len(new_data)} new items!")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.info("⏱️ Knowledge base is already up to date!")
        
        else:  # Load Sample Data
            if st.button("Load Sample Knowledge", type="primary"):
                texts, sources = DataSourceLoader.load_sample_knowledge()
                st.session_state.kb.add_documents(texts, sources)
                st.success(f"✅ Loaded {len(texts)} sample documents!")
                time.sleep(0.5)
                st.rerun()
        
        # View KB Contents
        with st.expander("🔍 View Knowledge Base"):
            documents = st.session_state.kb.get_all_documents(limit=20)
            if documents:
                for i, doc in enumerate(documents, 1):
                    st.markdown(f"""
                        <div class="kb-result">
                            <small><strong>Doc {i}</strong> | {doc['source']}</small><br>
                            {doc['text'][:200]}{'...' if len(doc['text']) > 200 else ''}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No documents in knowledge base yet.")
        
        # Clear KB
        if st.button("🗑️ Clear Knowledge Base", type="secondary"):
            if st.session_state.kb.get_stats()['total_documents'] > 0:
                st.session_state.kb.clear()
                st.session_state.messages = []
                st.success("✅ Knowledge base cleared!")
                time.sleep(0.5)
                st.rerun()
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>You:</strong><br>{msg['content']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sentiment
                    st.markdown(f"""
                        <span class="sentiment-badge {msg['sentiment'].lower()}">
                            {msg['sentiment']} ({msg['confidence']:.1%})
                        </span>
                    """, unsafe_allow_html=True)
                    
                    # Show KB results if any
                    if msg.get('kb_results'):
                        relevant_results = [r for r in msg['kb_results'] if r['relevance_score'] > 0.5]
                        if relevant_results:
                            st.markdown("**📚 Relevant Knowledge:**")
                            for result in relevant_results[:2]:
                                st.markdown(f"""
                                    <div class="kb-result">
                                        <small>Relevance: {result['relevance_score']:.1%} | Source: {result['source']}</small><br>
                                        {result['text'][:250]}
                                    </div>
                                """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-message bot-message">
                            <strong>Assistant:</strong><br>{msg['content']}
                        </div>
                    """, unsafe_allow_html=True)
        
        # Input area
        st.markdown("---")
        user_input = st.text_area(
            "Ask me anything:",
            height=100,
            placeholder="Example: What's your refund policy? or I'm having issues with my order..."
        )
        
        col_a, col_b, col_c = st.columns([1, 1, 3])
        with col_a:
            send_button = st.button("Send 📤", type="primary", use_container_width=True)
        with col_b:
            clear_button = st.button("Clear Chat 🗑️", use_container_width=True)
        
        if clear_button:
            st.session_state.messages = []
            st.rerun()
        
        # Process input
        if send_button and user_input:
            with st.spinner("Processing with ChromaDB..."):
                # Sentiment analysis
                sentiment_result = sentiment_analyzer(user_input)[0]
                sentiment = sentiment_result['label']
                confidence = sentiment_result['score']
                
                # Search knowledge base using ChromaDB
                kb_results = st.session_state.kb.search(user_input, k=5)
                
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "kb_results": kb_results,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Generate response
                bot_response = generate_hybrid_response(
                    sentiment, confidence, user_input, kb_results
                )
                
                # Add bot message
                st.session_state.messages.append({
                    "role": "bot",
                    "content": bot_response,
                    "timestamp": datetime.now().isoformat()
                })
                
                st.rerun()
        
        # Example queries
        if not st.session_state.messages:
            st.markdown("### 💡 Try these examples:")
            col1, col2, col3 = st.columns(3)
            
            examples = [
                ("What's your return policy?", col1),
                ("I'm frustrated with delivery times", col2),
                ("Tell me about payment options", col3)
            ]
            
            for example, col in examples:
                with col:
                    if st.button(example, use_container_width=True):
                        # Process example
                        sentiment_result = sentiment_analyzer(example)[0]
                        kb_results = st.session_state.kb.search(example, k=5)
                        
                        st.session_state.messages.append({
                            "role": "user",
                            "content": example,
                            "sentiment": sentiment_result['label'],
                            "confidence": sentiment_result['score'],
                            "kb_results": kb_results,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        bot_response = generate_hybrid_response(
                            sentiment_result['label'],
                            sentiment_result['score'],
                            example,
                            kb_results
                        )
                        
                        st.session_state.messages.append({
                            "role": "bot",
                            "content": bot_response,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        st.rerun()

if __name__ == "__main__":
    main()