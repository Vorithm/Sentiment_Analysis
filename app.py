import streamlit as st 
import pandas as pd
import re
import base64

# --- NLTK imports ---
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")
sia = SentimentIntensityAnalyzer()

# Set page configuration
st.set_page_config(
    page_title="NLTK Sentiment Analysis Pro",
    page_icon="🧠",
    layout="wide"
)

# Function to show header image with gradient highlight behind logo
def header_image(image_file):
    with open(image_file, "rb") as f:
        encoded_bg = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <div style="
            background-image: url('data:image/jpg;base64,{encoded_bg}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            padding: 50px 30px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <!-- Title Section Only -->
            <div style="display:flex; flex-direction: column; justify-content: center; align-items:center;">
                <h1 style='color:black; text-shadow:1px 1px 4px #000; margin:0; font-size:36px;'>
                    🧠 NLTK Sentiment Analysis 
                </h1>
            </div>
        </div>
    """, unsafe_allow_html=True)

header_image(
    "assets/background.jpg")


# Custom CSS
st.markdown("""
<style>
    .sub-header { 
        font-size: 1.5rem; 
        color: var(--primary-color); 
        margin-bottom: 1rem; 
        font-weight: 600; 
    }
    .positive-word { 
        color: #2E8B57; 
        font-weight: bold; 
        background-color: rgba(46, 139, 87, 0.15); 
        padding: 2px 6px; 
        border-radius: 4px; 
        margin: 2px; 
    }
    .negative-word { 
        color: #DC143C; 
        font-weight: bold; 
        background-color: rgba(220, 20, 60, 0.15); 
        padding: 2px 6px; 
        border-radius: 4px; 
        margin: 2px; 
    }
    .neutral-word { 
        color: var(--text-color); 
        font-weight: bold; 
        background-color: var(--secondary-background-color); 
        padding: 2px 6px; 
        border-radius: 4px; 
        margin: 2px; 
    }
    .analysis-container { 
        background-color: var(--secondary-background-color); 
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px; 
        border-left: 5px solid var(--primary-color); 
    }
            

    .stats-box { 
        background: linear-gradient(
            to bottom right, 
            rgba(255,255,255,0.15), 
            rgba(0,0,0,0.15)
        ); 
        padding: 15px; 
        border-radius: 12px; 
        margin: 10px 0; 
        backdrop-filter: blur(6px);  /* Glass effect */
        border: 1px solid rgba(255,255,255,0.25);
        color: var(--text-color);    /* Auto adjust text */
        text-align: center;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #ff7e5f, #feb47b, #86a8e7, #91eae4);
        background-size: 300% 100%;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        height: 50px;
        width: 100%;
        font-size: 18px;
        transition: 0.5s;
    }
    .stButton>button:hover {
        background-position: 100% 0;
        transform: scale(1.05);
    }
    textarea {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        padding: 10px !important;
        font-size: 16px !important;
        color: var(--text-color) !important;
        background: linear-gradient(
            to bottom right,
            rgba(255,255,255,0.15),
            rgba(0,0,0,0.15)
        ) !important;
        backdrop-filter: blur(6px) !important;
    }
    
        /* Glassmorphism Table */
    .stDataFrame {
        background: linear-gradient(
            to bottom right, 
            rgba(255,255,255,0.15), 
            rgba(0,0,0,0.15)
        ) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        backdrop-filter: blur(6px) !important;
        overflow: hidden;
    }
    
    /* Table ke andar ke cells transparent ho */
    .stDataFrame div {
        background: transparent !important;
        color: var(--text-color) !important;
    }

    
</style>
""", unsafe_allow_html=True)


# Preprocessing
def advanced_preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s\.\!\?\,]', '', text)
    return re.sub(r'\s+', ' ', text.lower()).strip()

# NLTK word-level sentiment
def analyze_words_nltk(text):
    words = advanced_preprocess_text(text).split()
    results = []
    for word in words:
        score = sia.polarity_scores(word)
        compound = score['compound']
        if compound >= 0.05:
            sentiment = 'POSITIVE'
        elif compound <= -0.05:
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'NEUTRAL'
        results.append({
            'word': word,
            'context': word,
            'scores': {
                'POSITIVE': max(compound, 0),
                'NEGATIVE': max(-compound, 0),
                'NEUTRAL': 1 - abs(compound)
            },
            'dominant_sentiment': sentiment,
            'confidence': abs(compound)
        })
    return results

# Generate colored HTML text
def generate_colored_text(results):
    output = []
    for r in results:
        word, sentiment, conf = r['word'], r['dominant_sentiment'], r['confidence']
        if sentiment == 'POSITIVE':
            output.append(f"<span class='positive-word' title='Confidence: {conf:.2%}'>{word}</span>")
        elif sentiment == 'NEGATIVE':
            output.append(f"<span class='negative-word' title='Confidence: {conf:.2%}'>{word}</span>")
        else:
            output.append(f"<span class='neutral-word' title='Confidence: {conf:.2%}'>{word}</span>")
    return ' '.join(output)

# Main app
def main():
    st.sidebar.markdown("<h2 style='text-align:center; color:#A23B72;'>⚙️ Settings Panel</h2>", unsafe_allow_html=True)
    min_confidence = st.sidebar.slider("🟢 Neutral Threshold", 0.0, 1.0, 0.3, 0.05)
    show_details = st.sidebar.checkbox("📋 Show Detailed Analysis", value=True)

    st.markdown("<h3 class='sub-header'>📝 Enter Text for Analysis</h3>", unsafe_allow_html=True)
    user_input = st.text_area("Input your text here:", height=150, placeholder="Type or paste your text here...")

    if st.button("🚀 Analyze", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter text.")
            return

        nltk_results = analyze_words_nltk(user_input)

        st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
        st.markdown("### 🎨 NLTK Word-Level Sentiment Analysis")
        st.markdown(generate_colored_text(nltk_results), unsafe_allow_html=True)

        counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
        for r in nltk_results:
            counts[r['dominant_sentiment']] += 1
        total = len(nltk_results)

        stat1, stat2, stat3 = st.columns(3)
        stat1.markdown(f"<div class='stats-box'><h4>Positive</h4><h2 style='color:#2E8B57'>{counts['POSITIVE']}</h2><small>{counts['POSITIVE']/total*100:.1f}%</small></div>", unsafe_allow_html=True)
        stat2.markdown(f"<div class='stats-box'><h4>Negative</h4><h2 style='color:#DC143C'>{counts['NEGATIVE']}</h2><small>{counts['NEGATIVE']/total*100:.1f}%</small></div>", unsafe_allow_html=True)
        stat3.markdown(f"<div class='stats-box'><h4>Neutral</h4><h2 style='color:#696969'>{counts['NEUTRAL']}</h2><small>{counts['NEUTRAL']/total*100:.1f}%</small></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Detailed table
        if show_details and nltk_results:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h3 '>🔍 Detailed Sentiment Analysis Table</h3>", unsafe_allow_html=True)
        
            table_data = []
            for r in nltk_results:
                if r['confidence'] >= min_confidence:
                    table_data.append({
                        'Word': r['word'],
                        'Positive': f"{r['scores']['POSITIVE']:.3f}",
                        'Negative': f"{r['scores']['NEGATIVE']:.3f}",
                        'Neutral': f"{r['scores']['NEUTRAL']:.3f}",
                        'Dominant': r['dominant_sentiment'],
                        'Confidence': f"{r['confidence']:.3f}"
                    })
        
            if table_data:
                df = pd.DataFrame(table_data)
        
                def highlight_sentiment(val):
                    if val == 'POSITIVE':
                        return 'background-color: rgba(46, 139, 87, 0.2); color: #2E8B57; font-weight: bold;'
                    elif val == 'NEGATIVE':
                        return 'background-color: rgba(220, 20, 60, 0.2); color: #DC143C; font-weight: bold;'
                    else:
                        return 'background-color: rgba(128, 128, 128, 0.2); color: var(--text-color); font-weight: bold;'
        
                styled_df = df.style.applymap(highlight_sentiment, subset=['Dominant'])
        
                st.dataframe(styled_df, use_container_width=True, height=400)
        
                # Extra CSS for hover effect
                st.markdown("""
                <style>
                .stDataFrame tbody tr:hover {
                    background-color: rgba(255,255,255,0.1) !important;
                    transform: scale(1.01);
                    transition: all 0.2s ease-in-out;
                }
                </style>
                """, unsafe_allow_html=True)
        
            else:
                st.info("No words meet the minimum confidence threshold.")


        # --- About Section ---
        st.markdown("---")
        st.markdown("""
        ### 🧠 About This NLTK Sentiment Analysis
        **Model Architecture:**
        - **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
        - Lightweight lexicon-based sentiment analysis tool
        - Works well on short texts, social media, and product reviews  

        **Key Features:**
        - ✅ Fast, no heavy model download
        - ✅ Word-level polarity detection
        - ✅ Overall sentiment score for input text
        - ✅ Sentiment distribution visualization  

        **Color Legend:**
        - 🟢 Green: Positive sentiment words  
        - 🔴 Red: Negative sentiment words  
        - ⚫ Gray: Neutral sentiment words  

        Hover over words to see compound scores!
        """)

       

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center;'>"
        "Vorithm © 2025 | NLP Sentiment Explorer"
        "</p>",
        unsafe_allow_html=True
    )
if __name__ == "__main__":

    main()
