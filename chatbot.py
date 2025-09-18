import re
import warnings
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
import pycountry

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

warnings.filterwarnings('ignore')
DetectorFactory.seed = 0

app = Flask(__name__)
CORS(app)

disease_info = {
    "Malaria": {
        "symptoms": "High fever, chills, headache, nausea, vomiting, muscle pain, fatigue, and sweating.",
        "prevention": "Use mosquito nets, insect repellent, wear long sleeves, eliminate standing water.",
        "first_aid": "Seek immediate medical attention. Hydrate, manage fever with paracetamol, rest.",
        "severity": "High",
        "category": "Infectious Disease"
    },
    # ... Add more diseases here ...
    "Dengue": {
        "symptoms": "High fever, severe headache, pain behind the eyes, rash, mild bleeding.",
        "prevention": "Prevent mosquito bites, eliminate mosquito breeding sites.",
        "first_aid": "Rest, hydrate, paracetamol for fever, consult doctor.",
        "severity": "High",
        "category": "Infectious Disease"
    },
}

def detect_language(text):
    try:
        lang_code = detect(text)
        lang_code = lang_code.split('-')[0].lower()
        hindi_keywords = ['hai', 'mera', 'mujhe', 'ka', 'se', 'kya', 'kaise']
        english_keywords = ['and', 'the', 'is', 'have', 'my', 'with', 'pain']
        text_lower = text.lower()
        has_hindi = any(word in text_lower for word in hindi_keywords)
        has_english = any(word in text_lower for word in english_keywords)

        if has_hindi and has_english:
            return 'hinglish', 'Hinglish'
        elif lang_code == 'hi' or has_hindi:
            return 'hi', 'Hindi'

        lang = pycountry.languages.get(alpha_2=lang_code)
        if lang:
            return lang_code, lang.name
        else:
            return lang_code, lang_code.upper()
    except:
        return 'en', 'English'

def translate_to_english(text, src_lang=None):
    if not text.strip():
        return ""
    if src_lang is None:
        src_lang, _ = detect_language(text)
    if src_lang == 'en':
        return text
    try:
        if src_lang in ['hinglish', 'hi']:
            return GoogleTranslator(source='hi', target='en').translate(text)
        return GoogleTranslator(source=src_lang, target='en').translate(text)
    except:
        return text

def translate_from_english(text, tgt_lang):
    if not text.strip() or tgt_lang == 'en':
        return text
    if tgt_lang == 'hinglish':
        tgt_lang = 'hi'
    try:
        return GoogleTranslator(source='en', target=tgt_lang).translate(text)
    except:
        return text

def process_multilingual_input(text):
    lang_code, lang_name = detect_language(text)
    en_text = translate_to_english(text, lang_code)
    return {
        'original_text': text,
        'original_lang_code': lang_code,
        'original_lang_name': lang_name,
        'english_text': en_text
    }

def clean_text(text):
    return re.sub(r'[^a-z\s]', '', text.lower())

class EnhancedSymptomChatbot:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.model = None
        self._train_model()

    def _train_model(self):
        texts = [
            "fever cough headache fatigue",
            "high fever chills headache nausea muscle pain",
            "severe headache joint pain rash fever bleeding",
            "sustained fever weakness stomach pain loss appetite",
            "frequent urination thirst weight loss fatigue blurred vision",
            "headache dizziness chest pain shortness breath",
            "runny nose sore throat cough sneezing mild fever",
            "severe headache nausea vomiting light sensitivity",
            "nausea vomiting diarrhea stomach cramps fever",
            "persistent cough mucus fatigue shortness breath chest pain",
            "fever chills muscle aches fatigue headache dry cough"
        ] * 5

        texts = [clean_text(t) for t in texts]
        labels = [
            "Flu", "Malaria", "Dengue", "Typhoid", "Diabetes", "Hypertension",
            "Common Cold", "Migraine", "Gastroenteritis", "Bronchitis",
            "Flu"
        ] * 5

        y = self.label_encoder.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1,3), max_features=1000),
            MultinomialNB(alpha=0.1)
        )
        self.model.fit(X_train, y_train)
        print("Model trained successfully!")

    def generate_chat_response(self, user_input, lang_code='en'):
        cleaned = clean_text(user_input)
        if not cleaned:
            reply = "Please enter some symptoms to help you."
            return {'reply': reply, 'detected_symptoms': [], 'prediction': None, 'confidence': 0}

        pred_idx = self.model.predict([cleaned])[0]
        prediction = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = max(self.model.predict_proba([cleaned])[0]) * 100

        # Simple symptom detection
        symptoms_map = {
            'fever': ['fever', 'temperature', 'hot'],
            'cough': ['cough', 'coughing'],
            'headache': ['headache', 'head pain', 'migraine'],
            'nausea': ['nausea', 'sick', 'queasy'],
            'vomiting': ['vomiting', 'throwing up', 'vomit'],
            'fatigue': ['tired', 'fatigue', 'exhausted', 'weakness'],
            'diarrhea': ['diarrhea', 'loose stool', 'stomach upset'],
            'rash': ['rash', 'skin irritation'],
            'joint pain': ['joint pain', 'body ache', 'muscle pain'],
            'chest pain': ['chest pain', 'breathing difficulty'],
            'dizziness': ['dizzy', 'dizziness', 'lightheaded'],
            'sore throat': ['sore throat', 'throat pain'],
            'runny nose': ['runny nose', 'sneezing', 'congestion']
        }

        detected_symptoms = [s for s,kwlist in symptoms_map.items() if any(k in cleaned for k in kwlist)]

        if detected_symptoms and confidence > 30:
            symptoms_str = ", ".join(detected_symptoms)
            if prediction in disease_info:
                info = disease_info[prediction]
                emoji = "🔴" if info['severity'] == 'High' else "🟡" if info['severity'] == 'Moderate' else "🟢"
                reply = f"{emoji} Based on your symptoms ({symptoms_str}), you may have {prediction} with {confidence:.2f}% confidence.\n" \
                        f"Category: {info['category']}\n" \
                        f"Severity: {info['severity']}\n" \
                        f"Symptoms: {info['symptoms']}\n" \
                        f"Prevention: {info['prevention']}\n" \
                        f"First Aid: {info['first_aid']}\n" \
                        "Please consult a healthcare professional for a definitive diagnosis."
            else:
                reply = f"Based on your symptoms ({symptoms_str}), you might have {prediction} with {confidence:.2f}% confidence. Please see a doctor."
        else:
            reply = "Sorry, I couldn't confidently identify your condition. Please describe your symptoms more specifically."

        if lang_code != 'en':
            reply = translate_from_english(reply, lang_code)

        return {
            'reply': reply,
            'detected_symptoms': detected_symptoms,
            'prediction': prediction,
            'confidence': confidence,
            'language': lang_code
        }

# Include mic and send button in frontend

FRONTEND_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Medical Chatbot with Mic</title>
    <style>
        /* Add your CSS styles here */
        #micBtn {
            cursor: pointer;
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 8px 12px;
            border-radius: 50%;
            margin-left: 8px;
            font-size: 18px;
        }
        #micBtn.recording {
            background-color: #f44336;
        }
    </style>
</head>
<body>
    <h2>AI Medical Chatbot</h2>
    <div id="chat"></div>
    <input type="text" id="input" placeholder="Describe your symptoms"/>
    <button id="sendBtn" onclick="send()">Send</button>
    <button id="micBtn" title="Click to speak">🎤</button>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('sendBtn');
        const micBtn = document.getElementById('micBtn');

        function addMessage(text, sender) {
            const p = document.createElement('p');
            p.textContent = (sender === 'user' ? 'You: ' : 'Bot: ') + text;
            p.style.color = sender === 'user' ? 'blue' : 'green';
            chat.appendChild(p);
            chat.scrollTop = chat.scrollHeight;
        }

        function send() {
            let text = input.value.trim();
            if(text === "") {
                alert("Please enter symptoms");
                return;
            }
            addMessage(text, 'user');
            input.value = "";
            sendBtn.disabled = true;
            micBtn.disabled = true;
            
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({message: text})
            })
            .then(resp => resp.json())
            .then(data => addMessage(data.reply,'bot'))
            .catch(() => addMessage("Error connecting to server",'bot'))
            .finally(() => {
                sendBtn.disabled = false;
                micBtn.disabled = false;
            });
        }

        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            micBtn.disabled = true;
            alert("Speech Recognition not supported by your browser.");
        } else {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            const recognition = new SpeechRecognition();

            recognition.lang = 'en-US';  // can be dynamic
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;

            recognition.onstart = () => micBtn.classList.add('recording');
            recognition.onend = () => micBtn.classList.remove('recording');
            recognition.onerror = (e) => alert("Speech recognition error: " + e.error);

            recognition.onresult = (event) => {
                input.value = event.results[0][0].transcript;
                input.dispatchEvent(new Event('input'));
            };

            micBtn.onclick = () => {
                if(micBtn.classList.contains('recording')) {
                    recognition.stop();
                } else {
                    recognition.start();
                }
            };
        }

        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                send();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(FRONTEND_HTML)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    text = data.get('message', '').strip()
    if not text:
        return jsonify({'error': 'No input text provided.'}), 400
    lang_data = process_multilingual_input(text)
    response = chatbot.generate_chat_response(lang_data['english_text'], lang_data['original_lang_code'])
    response['detected_language'] = lang_data['original_lang_name']
    return jsonify(response)

if __name__ == '__main__':
    chatbot = EnhancedSymptomChatbot()  # initialize & train model
    print("Starting AI Medical Chatbot with microphone...")
    app.run(debug=True, host='0.0.0.0', port=5000)
