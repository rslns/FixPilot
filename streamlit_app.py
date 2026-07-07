"""
AI-Powered Vehicle Assistant
-----------------------------
Helps vehicle owners diagnose issues via text description, get safe DIY
repair guidance, rough cost/tool estimates, and preventive maintenance tips.
Supports English, Hindi, Telugu, Tamil, and Kannada.
Built with Streamlit + the Google Gemini API.
"""

import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()  # reads .env into os.environ

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Vehicle Assistant",
    page_icon="🔧",
    layout="wide",
)

DEFAULT_MODEL = "gemini-3.5-flash"

# --------------------------------------------------------------------------
# Languages
# --------------------------------------------------------------------------
LANGUAGES = {
    "en": "English",
    "hi": "हिंदी (Hindi)",
    "te": "తెలుగు (Telugu)",
    "ta": "தமிழ் (Tamil)",
    "kn": "ಕನ್ನಡ (Kannada)",
}

LANGUAGE_NAME_FOR_PROMPT = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
}

UI_TEXT = {
    "en": {
        "title": "🔧 FixPilot",
        "caption": "Guiding users through repairs",
        "language_label": "Choose your language",
        "vehicle_header": "Your vehicle",
        "type_label": "Type",
        "type_options": ["Car", "Motorcycle/Bike"],
        "make_label": "Make",
        "make_placeholder": "e.g. Honda",
        "model_label": "Model",
        "model_placeholder": "e.g. Civic",
        "year_label": "Year",
        "year_placeholder": "e.g. 2018",
        "mileage_label": "Mileage/Odometer",
        "mileage_placeholder": "e.g. 45000 km",
        "currency_label": "Currency for cost estimates",
        "clear_button": "Clear conversation",
        "vehicle_disclaimer": "⚠️ This tool gives general guidance only and is not a substitute for a hands-on inspection by a certified mechanic, especially for brakes, steering, airbags, or fuel-system issues.",
        "tab_diagnose": "Diagnose an issue",
        "tab_maintenance": "Preventive maintenance",
        "tab_about": "About / Disclaimer",
        "chat_placeholder": "Describe the problem (e.g. 'grinding noise when I brake')",
        "maintenance_header": "Get a preventive maintenance checklist",
        "maintenance_desc": "Uses the vehicle details entered above. Fill those in first for a tailored checklist.",
        "maintenance_button": "Generate maintenance checklist",
        "about_what_header": "What this does",
        "about_what_body": (
            "- **Text diagnosis** — describe a symptom (noise, smell, warning light, handling issue) and get "
            "likely causes, urgency level, DIY steps (when safe), tools needed, and a rough cost estimate.\n"
            "- **Preventive maintenance** — generate a checklist tailored to your vehicle's age and mileage."
        ),
        "about_disclaimer_header": "Disclaimer",
        "about_disclaimer_body": (
            "This assistant provides general, AI-generated guidance based on the information you supply. "
            "It is **not** a certified mechanic and cannot physically inspect your vehicle. For brakes, "
            "steering, airbags, fuel-system, or structural issues — or any time the assistant flags 🔴 "
            "Critical — stop driving and consult a qualified professional before continuing to use the "
            "vehicle. Cost estimates are rough approximations and vary by region, vehicle, and shop."
        ),
        "session_started": "Session started",
    },
    "hi": {
        "title": "🔧 व्हीकल असिस्टेंट",
        "caption": "समस्या का वर्णन करें और सरल भाषा में निदान, DIY चरण, और लागत अनुमान प्राप्त करें।",
        "language_label": "🌐 अपनी भाषा चुनें",
        "vehicle_header": "आपका वाहन",
        "type_label": "प्रकार",
        "type_options": ["कार", "मोटरसाइकिल/बाइक"],
        "make_label": "कंपनी (Make)",
        "make_placeholder": "जैसे Honda",
        "model_label": "मॉडल",
        "model_placeholder": "जैसे Civic",
        "year_label": "वर्ष",
        "year_placeholder": "जैसे 2018",
        "mileage_label": "माइलेज/ओडोमीटर",
        "mileage_placeholder": "जैसे 45000 km",
        "currency_label": "लागत अनुमान के लिए मुद्रा",
        "clear_button": "🗑️ बातचीत साफ़ करें",
        "vehicle_disclaimer": "⚠️ यह टूल केवल सामान्य मार्गदर्शन देता है और प्रमाणित मैकेनिक द्वारा प्रत्यक्ष जांच का विकल्प नहीं है, विशेष रूप से ब्रेक, स्टीयरिंग, एयरबैग या ईंधन-प्रणाली से जुड़ी समस्याओं के लिए।",
        "tab_diagnose": "🩺 समस्या का निदान करें",
        "tab_maintenance": "🛠️ निवारक रखरखाव",
        "tab_about": "ℹ️ जानकारी / अस्वीकरण",
        "chat_placeholder": "समस्या बताएं (जैसे 'ब्रेक लगाने पर घर्षण जैसी आवाज़ आती है')",
        "maintenance_header": "निवारक रखरखाव सूची प्राप्त करें",
        "maintenance_desc": "यह ऊपर दी गई वाहन जानकारी का उपयोग करता है। सटीक सूची के लिए पहले वह जानकारी भरें।",
        "maintenance_button": "रखरखाव सूची बनाएं",
        "about_what_header": "यह क्या करता है",
        "about_what_body": (
            "- **टेक्स्ट निदान** — लक्षण बताएं (आवाज़, गंध, चेतावनी लाइट, हैंडलिंग समस्या) और संभावित कारण, "
            "तात्कालिकता स्तर, DIY चरण (जब सुरक्षित हो), आवश्यक उपकरण, और अनुमानित लागत प्राप्त करें।\n"
            "- **निवारक रखरखाव** — अपने वाहन की उम्र और माइलेज के अनुसार एक सूची बनाएं।"
        ),
        "about_disclaimer_header": "अस्वीकरण",
        "about_disclaimer_body": (
            "यह असिस्टेंट आपके द्वारा दी गई जानकारी के आधार पर सामान्य, एआई-जनित मार्गदर्शन प्रदान करता है। "
            "यह प्रमाणित मैकेनिक **नहीं** है और आपके वाहन का प्रत्यक्ष निरीक्षण नहीं कर सकता। ब्रेक, स्टीयरिंग, "
            "एयरबैग, ईंधन-प्रणाली, या संरचनात्मक समस्याओं के लिए — या जब भी असिस्टेंट 🔴 गंभीर चेतावनी दे — "
            "वाहन चलाना बंद करें और किसी योग्य पेशेवर से सलाह लें। लागत अनुमान क्षेत्र, वाहन और दुकान के अनुसार बदल सकते हैं।"
        ),
        "session_started": "सत्र प्रारंभ",
    },
    "te": {
        "title": "🔧 వాహన సహాయకుడు",
        "caption": "సమస్యను వివరించండి మరియు సాధారణ భాషలో నిర్ధారణ, DIY దశలు, ఖర్చు అంచనాలు పొందండి.",
        "language_label": "🌐 మీ భాషను ఎంచుకోండి",
        "vehicle_header": "మీ వాహనం",
        "type_label": "రకం",
        "type_options": ["కారు", "మోటార్‌సైకిల్/బైక్"],
        "make_label": "తయారీదారు (Make)",
        "make_placeholder": "ఉదా. Honda",
        "model_label": "మోడల్",
        "model_placeholder": "ఉదా. Civic",
        "year_label": "సంవత్సరం",
        "year_placeholder": "ఉదా. 2018",
        "mileage_label": "మైలేజ్/ఓడోమీటర్",
        "mileage_placeholder": "ఉదా. 45000 km",
        "currency_label": "ఖర్చు అంచనా కరెన్సీ",
        "clear_button": "🗑️ సంభాషణ తీసివేయండి",
        "vehicle_disclaimer": "⚠️ ఈ టూల్ సాధారణ మార్గదర్శకత్వం మాత్రమే ఇస్తుంది, ధృవీకరించబడిన మెకానిక్ ప్రత్యక్ష తనిఖీకి ప్రత్యామ్నాయం కాదు, ముఖ్యంగా బ్రేకులు, స్టీరింగ్, ఎయిర్‌బ్యాగ్‌లు లేదా ఇంధన వ్యవస్థ సమస్యలకు.",
        "tab_diagnose": "🩺 సమస్యను నిర్ధారించండి",
        "tab_maintenance": "🛠️ నివారణ నిర్వహణ",
        "tab_about": "ℹ️ వివరాలు / నిరాకరణ",
        "chat_placeholder": "సమస్యను వివరించండి (ఉదా. 'బ్రేక్ వేసినప్పుడు రుద్దుకునే శబ్దం వస్తుంది')",
        "maintenance_header": "నివారణ నిర్వహణ చెక్‌లిస్ట్ పొందండి",
        "maintenance_desc": "పైన ఇచ్చిన వాహన వివరాలను ఉపయోగిస్తుంది. ఖచ్చితమైన చెక్‌లిస్ట్ కోసం ముందుగా అవి పూరించండి.",
        "maintenance_button": "నిర్వహణ చెక్‌లిస్ట్ తయారు చేయండి",
        "about_what_header": "ఇది ఏమి చేస్తుంది",
        "about_what_body": (
            "- **టెక్స్ట్ నిర్ధారణ** — లక్షణాన్ని వివరించండి (శబ్దం, వాసన, హెచ్చరిక లైట్, హ్యాండ్లింగ్ సమస్య) మరియు సంభావ్య "
            "కారణాలు, తీవ్రత స్థాయి, DIY దశలు (సురక్షితమైనప్పుడు), అవసరమైన పరికరాలు, అంచనా ఖర్చు పొందండి.\n"
            "- **నివారణ నిర్వహణ** — మీ వాహనం వయస్సు మరియు మైలేజ్ ఆధారంగా చెక్‌లిస్ట్ తయారు చేస్తుంది."
        ),
        "about_disclaimer_header": "నిరాకరణ",
        "about_disclaimer_body": (
            "ఈ సహాయకుడు మీరు అందించిన సమాచారం ఆధారంగా సాధారణ, AI-ఆధారిత మార్గదర్శకత్వాన్ని అందిస్తుంది. ఇది "
            "ధృవీకరించబడిన మెకానిక్ **కాదు** మరియు మీ వాహనాన్ని ప్రత్యక్షంగా పరిశీలించలేదు. బ్రేకులు, స్టీరింగ్, "
            "ఎయిర్‌బ్యాగ్‌లు, ఇంధన వ్యవస్థ, లేదా నిర్మాణ సమస్యల కోసం — లేదా సహాయకుడు 🔴 తీవ్రమైనది అని సూచించినప్పుడు — "
            "వాహనం నడపడం ఆపి, అర్హత కలిగిన నిపుణుడిని సంప్రదించండి. ఖర్చు అంచనాలు ప్రాంతం, వాహనం, దుకాణాన్ని బట్టి మారుతాయి."
        ),
        "session_started": "సెషన్ ప్రారంభమైంది",
    },
    "ta": {
        "title": "🔧 வாகன உதவியாளர்",
        "caption": "பிரச்சினையை விவரிக்கவும், எளிய மொழியில் நோய் கண்டறிதல், DIY படிகள், செலவு மதிப்பீடுகளைப் பெறவும்.",
        "language_label": "🌐 உங்கள் மொழியைத் தேர்ந்தெடுக்கவும்",
        "vehicle_header": "உங்கள் வாகனம்",
        "type_label": "வகை",
        "type_options": ["கார்", "மோட்டார் சைக்கிள்/பைக்"],
        "make_label": "தயாரிப்பாளர் (Make)",
        "make_placeholder": "எ.கா. Honda",
        "model_label": "மாடல்",
        "model_placeholder": "எ.கா. Civic",
        "year_label": "ஆண்டு",
        "year_placeholder": "எ.கா. 2018",
        "mileage_label": "மைலேஜ்/ஓடோமீட்டர்",
        "mileage_placeholder": "எ.கா. 45000 km",
        "currency_label": "செலவு மதிப்பீட்டிற்கான நாணயம்",
        "clear_button": "🗑️ உரையாடலை அழிக்கவும்",
        "vehicle_disclaimer": "⚠️ இந்த கருவி பொதுவான வழிகாட்டுதலை மட்டுமே வழங்குகிறது, சான்றளிக்கப்பட்ட மெக்கானிக் நேரடி ஆய்வுக்கு மாற்றாக அல்ல, குறிப்பாக பிரேக், ஸ்டீயரிங், ஏர்பேக் அல்லது எரிபொருள் அமைப்பு சிக்கல்களுக்கு.",
        "tab_diagnose": "🩺 பிரச்சினையை கண்டறியவும்",
        "tab_maintenance": "🛠️ முன்னெச்சரிக்கை பராமரிப்பு",
        "tab_about": "ℹ️ விவரம் / மறுப்பு",
        "chat_placeholder": "பிரச்சினையை விவரிக்கவும் (எ.கா. 'பிரேக் போடும்போது தேய்க்கும் சத்தம்')",
        "maintenance_header": "முன்னெச்சரிக்கை பராமரிப்பு பட்டியலைப் பெறவும்",
        "maintenance_desc": "மேலே கொடுக்கப்பட்ட வாகன விவரங்களைப் பயன்படுத்துகிறது. துல்லியமான பட்டியலுக்கு முதலில் அவற்றை நிரப்பவும்.",
        "maintenance_button": "பராமரிப்பு பட்டியலை உருவாக்கவும்",
        "about_what_header": "இது என்ன செய்கிறது",
        "about_what_body": (
            "- **உரை நோய் கண்டறிதல்** — அறிகுறியை விவரிக்கவும் (சத்தம், வாசனை, எச்சரிக்கை விளக்கு, கையாளுதல் சிக்கல்) "
            "மற்றும் சாத்தியமான காரணங்கள், அவசர நிலை, DIY படிகள் (பாதுகாப்பாக இருக்கும்போது), தேவையான கருவிகள், "
            "தோராயமான செலவு மதிப்பீட்டைப் பெறவும்.\n"
            "- **முன்னெச்சரிக்கை பராமரிப்பு** — உங்கள் வாகனத்தின் வயது மற்றும் மைலேஜ் அடிப்படையில் பட்டியலை உருவாக்குகிறது."
        ),
        "about_disclaimer_header": "மறுப்பு",
        "about_disclaimer_body": (
            "இந்த உதவியாளர் நீங்கள் வழங்கும் தகவலின் அடிப்படையில் பொதுவான, AI-உருவாக்கிய வழிகாட்டுதலை வழங்குகிறது. "
            "இது சான்றளிக்கப்பட்ட மெக்கானிக் **அல்ல** மற்றும் உங்கள் வாகனத்தை நேரடியாக ஆய்வு செய்ய முடியாது. பிரேக், "
            "ஸ்டீயரிங், ஏர்பேக், எரிபொருள் அமைப்பு, அல்லது கட்டமைப்பு சிக்கல்களுக்கு — அல்லது உதவியாளர் 🔴 கடுமையானது "
            "எனக் குறிப்பிடும் போது — வாகனத்தை ஓட்டுவதை நிறுத்தி, தகுதியான நிபுணரை அணுகவும். செலவு மதிப்பீடுகள் "
            "பகுதி, வாகனம், கடை என மாறுபடும்."
        ),
        "session_started": "அமர்வு தொடங்கியது",
    },
    "kn": {
        "title": "🔧 ವಾಹನ ಸಹಾಯಕ",
        "caption": "ಸಮಸ್ಯೆಯನ್ನು ವಿವರಿಸಿ ಮತ್ತು ಸರಳ ಭಾಷೆಯಲ್ಲಿ ರೋಗನಿರ್ಣಯ, DIY ಹಂತಗಳು, ವೆಚ್ಚ ಅಂದಾಜುಗಳನ್ನು ಪಡೆಯಿರಿ.",
        "language_label": "🌐 ನಿಮ್ಮ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "vehicle_header": "ನಿಮ್ಮ ವಾಹನ",
        "type_label": "ಪ್ರಕಾರ",
        "type_options": ["ಕಾರು", "ಮೋಟಾರ್‌ಸೈಕಲ್/ಬೈಕ್"],
        "make_label": "ತಯಾರಕ (Make)",
        "make_placeholder": "ಉದಾ. Honda",
        "model_label": "ಮಾದರಿ",
        "model_placeholder": "ಉದಾ. Civic",
        "year_label": "ವರ್ಷ",
        "year_placeholder": "ಉದಾ. 2018",
        "mileage_label": "ಮೈಲೇಜ್/ಓಡೋಮೀಟರ್",
        "mileage_placeholder": "ಉದಾ. 45000 km",
        "currency_label": "ವೆಚ್ಚ ಅಂದಾಜಿಗೆ ಕರೆನ್ಸಿ",
        "clear_button": "🗑️ ಸಂಭಾಷಣೆ ತೆರವುಗೊಳಿಸಿ",
        "vehicle_disclaimer": "⚠️ ಈ ಟೂಲ್ ಸಾಮಾನ್ಯ ಮಾರ್ಗದರ್ಶನವನ್ನು ಮಾತ್ರ ನೀಡುತ್ತದೆ, ಪ್ರಮಾಣೀಕೃತ ಮೆಕానిక్ ಪ್ರತ್ಯಕ್ಷ ಪರಿಶೀಲನೆಗೆ ಪರ್ಯಾಯವಲ್ಲ, ವಿಶೇಷವಾಗಿ ಬ್ರೇಕ್, ಸ್ಟೀಯರಿಂಗ್, ಏರ್‌ಬ್ಯಾಗ್ ಅಥವಾ ಇಂಧನ ವ್ಯವಸ್ಥೆ ಸಮಸ್ಯೆಗಳಿಗೆ.",
        "tab_diagnose": "🩺 ಸಮಸ್ಯೆಯನ್ನು ಪತ್ತೆಹಚ್ಚಿ",
        "tab_maintenance": "🛠️ ತಡೆಗಟ್ಟುವ ನಿರ್ವಹಣೆ",
        "tab_about": "ℹ️ ಮಾಹಿತಿ / ಹಕ್ಕುತ್ಯಾಗ",
        "chat_placeholder": "ಸಮಸ್ಯೆಯನ್ನು ವಿವರಿಸಿ (ಉದಾ. 'ಬ್ರೇಕ್ ಹಾಕಿದಾಗ ಉಜ್ಜುವ ಶಬ್ದ ಬರುತ್ತದೆ')",
        "maintenance_header": "ತಡೆಗಟ್ಟುವ ನಿರ್ವಹಣೆ ಪಟ್ಟಿ ಪಡೆಯಿರಿ",
        "maintenance_desc": "ಮೇಲೆ ನೀಡಿದ ವಾಹನ ವಿವರಗಳನ್ನು ಬಳಸುತ್ತದೆ. ನಿಖರವಾದ ಪಟ್ಟಿಗಾಗಿ ಮೊದಲು ಅವುಗಳನ್ನು ಭರ್ತಿ ಮಾಡಿ.",
        "maintenance_button": "ನಿರ್ವಹಣೆ ಪಟ್ಟಿ ರಚಿಸಿ",
        "about_what_header": "ಇದು ಏನು ಮಾಡುತ್ತದೆ",
        "about_what_body": (
            "- **ಪಠ್ಯ ರೋಗನಿರ್ಣಯ** — ಲಕ್ಷಣವನ್ನು ವಿವರಿಸಿ (ಶಬ್ದ, ವಾಸನೆ, ಎಚ್ಚರಿಕೆ ದೀಪ, ಹ್ಯಾಂಡ್ಲಿಂಗ್ ಸಮಸ್ಯೆ) ಮತ್ತು "
            "ಸಂಭಾವ್ಯ ಕಾರಣಗಳು, ತುರ್ತು ಮಟ್ಟ, DIY ಹಂತಗಳು (ಸುರಕ್ಷಿತವಾಗಿದ್ದಾಗ), ಅಗತ್ಯ ಉಪಕರಣಗಳು, ಅಂದಾಜು ವೆಚ್ಚ ಪಡೆಯಿರಿ.\n"
            "- **ತಡೆಗಟ್ಟುವ ನಿರ್ವಹಣೆ** — ನಿಮ್ಮ ವಾಹನದ ವಯಸ್ಸು ಮತ್ತು ಮೈಲೇಜ್ ಆಧಾರದ ಮೇಲೆ ಪಟ್ಟಿಯನ್ನು ರಚಿಸುತ್ತದೆ."
        ),
        "about_disclaimer_header": "ಹಕ್ಕುತ್ಯಾಗ",
        "about_disclaimer_body": (
            "ಈ ಸಹಾಯಕವು ನೀವು ಒದಗಿಸಿದ ಮಾಹಿತಿಯ ಆಧಾರದ ಮೇಲೆ ಸಾಮಾನ್ಯ, AI-ರಚಿತ ಮಾರ್ಗದರ್ಶನವನ್ನು ನೀಡುತ್ತದೆ. ಇದು "
            "ಪ್ರಮಾಣೀಕೃತ ಮೆಕానిక్ **ಅಲ್ಲ** ಮತ್ತು ನಿಮ್ಮ ವಾಹನವನ್ನು ಪ್ರತ್ಯಕ್ಷವಾಗಿ ಪರಿಶೀಲಿಸಲಾಗುವುದಿಲ್ಲ. ಬ್ರೇಕ್, "
            "ಸ್ಟೀಯರಿಂಗ್, ಏರ್‌ಬ್ಯಾಗ್, ಇಂಧನ ವ್ಯವಸ್ಥೆ, ಅಥವಾ ರಚನಾತ್ಮಕ ಸಮಸ್ಯೆಗಳಿಗೆ — ಅಥವಾ ಸಹಾಯಕ 🔴 ಗಂಭೀರ ಎಂದು "
            "ಸೂಚಿಸಿದಾಗ — ವಾಹನ ಚಾಲನೆ ನಿಲ್ಲಿಸಿ, ಅರ್ಹ ವೃತ್ತಿಪರರನ್ನು ಸಂಪರ್ಕಿಸಿ. ವೆಚ್ಚ ಅಂದಾಜುಗಳು ಪ್ರದೇಶ, ವಾಹನ, "
            "ಅಂಗಡಿಯ ಆಧಾರದ ಮೇಲೆ ಬದಲಾಗುತ್ತವೆ."
        ),
        "session_started": "ಸೆಷನ್ ಪ್ರಾರಂಭವಾಗಿದೆ",
    },
}

# --------------------------------------------------------------------------
# Black theme with Royal Blue + Crimson accents
# --------------------------------------------------------------------------
ROYAL_BLUE = "#3B5BDB"
ROYAL_BLUE_DARK = "#1E3A8A"
CRIMSON = "#DC143C"
CRIMSON_DARK = "#B0102F"
BG_BLACK = "#0B0D12"
PANEL_BLACK = "#161922"
TEXT_LIGHT = "#E8E9EE"
TEXT_MUTED = "#9AA0B4"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {BG_BLACK};
            color: {TEXT_LIGHT};
        }}
        p, span, label, div, li {{
            color: {TEXT_LIGHT};
        }}
        h1, h2, h3 {{
            color: {ROYAL_BLUE} !important;
            font-weight: 800 !important;
        }}
        h1 {{
            border-bottom: 4px solid {CRIMSON};
            padding-bottom: 0.3rem;
            display: inline-block;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {PANEL_BLACK};
            border: 2px solid {ROYAL_BLUE} !important;
            border-radius: 14px;
            padding: 1rem 0.5rem;
            margin-bottom: 1.5rem;
        }}
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] div,
        div[data-baseweb="textarea"] textarea {{
            background-color: #1E212C !important;
            color: {TEXT_LIGHT} !important;
            border: 1px solid #2E3242 !important;
        }}
        div[data-testid="stWidgetLabel"] p {{
            color: {TEXT_LIGHT} !important;
            font-weight: 600;
        }}
        .clear-btn-spacer {{
            margin-top: 1.9rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {PANEL_BLACK};
            border-radius: 8px 8px 0 0;
            color: {TEXT_LIGHT};
            font-weight: 600;
            padding: 10px 18px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {ROYAL_BLUE} !important;
            color: white !important;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: {CRIMSON} !important;
        }}
        .stTabs [data-baseweb="tab-panel"] {{
            color: {TEXT_LIGHT};
        }}
        .stButton > button {{
            background-color: {CRIMSON};
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 700;
            padding: 0.5rem 1.2rem;
            width: 100%;
            transition: background-color 0.2s ease-in-out;
        }}
        .stButton > button:hover {{
            background-color: {CRIMSON_DARK};
            color: white;
        }}
        div[data-testid="stChatMessage"] {{
            background-color: {PANEL_BLACK};
            border-radius: 14px;
            padding: 10px 14px;
            margin-bottom: 8px;
        }}
        div[data-testid="stChatMessageAvatarUser"] {{
            background-color: {CRIMSON} !important;
        }}
        div[data-testid="stChatMessageAvatarAssistant"] {{
            background-color: {ROYAL_BLUE} !important;
        }}
        div[data-testid="stChatInput"] {{
            background-color: {PANEL_BLACK};
        }}
        div[data-testid="stChatInput"] textarea {{
            background-color: #1E212C !important;
            color: {TEXT_LIGHT} !important;
            border: 2px solid {ROYAL_BLUE} !important;
            border-radius: 10px !important;
            min-height: 90px !important;
            font-size: 16px !important;
            padding: 14px !important;
        }}
        div[data-testid="stAlert"] {{
            background-color: {PANEL_BLACK};
            border-left: 6px solid {CRIMSON};
            border-radius: 8px;
            color: {TEXT_LIGHT};
        }}
        .stCaption, small {{
            color: {TEXT_MUTED} !important;
        }}
        table {{
            color: {TEXT_LIGHT};
        }}
        thead tr th {{
            background-color: {ROYAL_BLUE_DARK} !important;
            color: white !important;
        }}
        tbody tr:nth-child(even) {{
            background-color: {PANEL_BLACK};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_SYSTEM_PROMPT = """You are "Vehicle Assistant", an AI mechanic-in-your-pocket that helps everyday
car and motorcycle owners understand what's wrong with their vehicle and make informed decisions.

You are talking to a non-technical vehicle owner, not a mechanic. Always:

1. Give your most likely diagnosis/diagnoses, ranked by probability, in plain language.
2. Rate the URGENCY as one of: 🟢 Low (can wait), 🟡 Medium (schedule soon), 🟠 High (fix within days),
   or 🔴 Critical/Safety issue (stop driving / see a professional immediately).
3. If it is genuinely safe for an average person with basic tools to attempt: give a clear,
   numbered, step-by-step DIY guide, plus a list of tools/parts needed.
4. If it is NOT safe or realistic for DIY (brakes, airbags, structural, electrical/fuel system
   risk, etc.), say so plainly and explain why, and recommend a professional instead of giving
   repair steps.
5. Give a rough, clearly-labeled cost estimate range (parts + typical labor if done by a shop),
   and note that costs vary heavily by region, vehicle, and shop.
6. Mention 1-2 relevant preventive maintenance tips related to the issue.
7. Never claim certainty. Remind the user, briefly, that a hands-on inspection by a qualified
   mechanic is the only way to be 100% sure, especially for safety-critical systems (brakes,
   steering, airbags, fuel, structural damage).

Format your answer in clean markdown with short headers so it's easy to scan on a phone.
Keep tone friendly, direct, and jargon-free (briefly define any technical term you must use).
Never fabricate a diagnosis with false confidence -- if the description is too vague, ask a
short, specific clarifying question instead of guessing wildly.
"""

BASE_MAINTENANCE_PROMPT = """You are "Vehicle Assistant", generating a preventive maintenance
checklist for an everyday vehicle owner (not a mechanic). Given the vehicle's type, make, model,
age, and mileage, produce:

1. A markdown table of maintenance items due now / soon / later, with rough recommended intervals.
2. A short list of warning signs the owner should never ignore for this vehicle type.
3. 2-3 money-saving preventive habits specific to this vehicle.

Keep it concise, practical, and in plain language.
"""


def with_language_instruction(base_prompt: str, lang_code: str) -> str:
    """Append a firm language instruction to any system prompt."""
    lang_name = LANGUAGE_NAME_FOR_PROMPT.get(lang_code, "English")
    if lang_code == "en":
        return base_prompt
    instruction = (
        f"\n\nIMPORTANT: Respond ENTIRELY in {lang_name}, written in the native {lang_name} script "
        f"(not transliterated / not Romanized). You may keep vehicle brand names, model names, and "
        f"widely-used technical terms (like 'ABS', 'RPM', 'brake pad') in English where that is the "
        f"natural convention in everyday {lang_name} speech, but all explanations, headers, and "
        f"instructions must be in {lang_name}."
    )
    return base_prompt + instruction


# --------------------------------------------------------------------------
# API key — pulled from .env / environment, never shown in the UI
# --------------------------------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vehicle" not in st.session_state:
    st.session_state.vehicle = {
        "type_idx": 0,
        "make": "",
        "model": "",
        "year": "",
        "mileage": "",
        "currency": "USD",
    }

if "lang" not in st.session_state:
    st.session_state.lang = "en"

if "last_error" not in st.session_state:
    st.session_state.last_error = None

# --------------------------------------------------------------------------
# Title (centered) — uses last-selected language, or English on first load
# --------------------------------------------------------------------------
T_current = UI_TEXT[st.session_state.lang]

st.markdown(
    f"""
    <h1 style="text-align: center; width: 100%; display: block; border-bottom: none;
               font-size: 3.5rem; margin-bottom: 0.2rem;">
        {T_current['title']}
    </h1>
    <p style="text-align: center; color: {TEXT_MUTED}; margin-top: -0.3rem; font-size: 1.1rem;">
        {T_current['caption']}
    </p>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Language selector (below title, drives everything below it)
# --------------------------------------------------------------------------
lang_display_to_code = {v: k for k, v in LANGUAGES.items()}
col_a, col_b, col_c = st.columns([1, 1, 1])
with col_b:
    selected_display = st.selectbox(
        T_current["language_label"],
        list(LANGUAGES.values()),
        index=list(LANGUAGES.keys()).index(st.session_state.lang),
        key="language_selector",
    )
st.session_state.lang = lang_display_to_code[selected_display]
T = UI_TEXT[st.session_state.lang]  # shorthand for current translation dict

# --------------------------------------------------------------------------
# Vehicle details (main page, no sidebar)
# --------------------------------------------------------------------------
with st.container(border=True):
    st.subheader(T["vehicle_header"])

    row1 = st.columns([1, 1, 1, 1])
    type_choice = row1[0].radio(
        T["type_label"], T["type_options"],
        index=st.session_state.vehicle["type_idx"],
    )
    st.session_state.vehicle["type_idx"] = T["type_options"].index(type_choice)
    st.session_state.vehicle["make"] = row1[1].text_input(
        T["make_label"], st.session_state.vehicle["make"], placeholder=T["make_placeholder"]
    )
    st.session_state.vehicle["model"] = row1[2].text_input(
        T["model_label"], st.session_state.vehicle["model"], placeholder=T["model_placeholder"]
    )
    st.session_state.vehicle["year"] = row1[3].text_input(
        T["year_label"], st.session_state.vehicle["year"], placeholder=T["year_placeholder"]
    )

    row2 = st.columns([1, 1, 1])
    st.session_state.vehicle["mileage"] = row2[0].text_input(
        T["mileage_label"], st.session_state.vehicle["mileage"], placeholder=T["mileage_placeholder"]
    )
    st.session_state.vehicle["currency"] = row2[1].selectbox(
        T["currency_label"], ["USD", "INR", "EUR", "GBP"],
        index=["USD", "INR", "EUR", "GBP"].index(
            st.session_state.vehicle["currency"])
    )
    with row2[2]:
        st.markdown('<div class="clear-btn-spacer">', unsafe_allow_html=True)
        if st.button(T["clear_button"], use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_error = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.caption(T["vehicle_disclaimer"])

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def vehicle_context_str():
    v = st.session_state.vehicle
    # keep English for the model
    vehicle_type_en = ["Car", "Motorcycle/Bike"][v["type_idx"]]
    parts = [f"Vehicle type: {vehicle_type_en}"]
    if v["make"]:
        parts.append(f"Make: {v['make']}")
    if v["model"]:
        parts.append(f"Model: {v['model']}")
    if v["year"]:
        parts.append(f"Year: {v['year']}")
    if v["mileage"]:
        parts.append(f"Mileage: {v['mileage']}")
    parts.append(f"Preferred currency for cost estimates: {v['currency']}")
    return " | ".join(parts)


def get_client():
    if not api_key:
        st.error(
            "GEMINI_API_KEY not found. Make sure it's set in your .env file (or Streamlit Secrets).")
        st.stop()
    return genai.Client(api_key=api_key)


def call_gemini(user_parts, base_system_prompt, history=None, max_tokens=None):
    """Send a message (with optional history) to Gemini and return the text reply.
    On failure, stores the error in st.session_state.last_error and returns None."""
    client = get_client()
    system_prompt = with_language_instruction(
        base_system_prompt, st.session_state.lang)

    if max_tokens is None:
        max_tokens = 6000 if st.session_state.lang != "en" else 3000

    contents = []
    if history:
        for m in history:
            contents.append(types.Content(role=m["role"], parts=m["parts"]))
    contents.append(types.Content(role="user", parts=user_parts))

    try:
        with st.spinner("Thinking..."):
            resp = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                ),
            )
        text = resp.text
        if not text:
            st.session_state.last_error = (
                "Gemini returned an empty response. This can happen if the request was "
                "blocked by safety filters, or if the model produced no output. "
                "Try rephrasing your message."
            )
            return None
        try:
            finish_reason = resp.candidates[0].finish_reason
            if str(finish_reason) == "MAX_TOKENS":
                text += "\n\n*(⚠️ Response was cut off — you can ask me to continue.)*"
        except Exception:
            pass
        st.session_state.last_error = None
        return text
    except APIError as e:
        st.session_state.last_error = f"API error: {e}"
        return None
    except Exception as e:
        st.session_state.last_error = f"Unexpected error: {e}"
        return None


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_diagnose, tab_maintenance, tab_about = st.tabs(
    [T["tab_diagnose"], T["tab_maintenance"], T["tab_about"]]
)

# ---------------- Diagnose tab ----------------
with tab_diagnose:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["display"])

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    user_text = st.chat_input(T["chat_placeholder"])

    if user_text:
        full_prompt = f"{vehicle_context_str()}\n\nOwner's description: {user_text}"
        parts = [types.Part.from_text(text=full_prompt)]

        st.session_state.messages.append({
            "role": "user",
            "parts": parts,
            "display": user_text,
        })

        reply = call_gemini(
            parts,
            BASE_SYSTEM_PROMPT,
            history=st.session_state.messages[:-1],
        )

        if reply:
            reply_part = types.Part.from_text(text=reply)
            st.session_state.messages.append({
                "role": "model",
                "parts": [reply_part],
                "display": reply,
            })
            st.rerun()
        else:
            # Remove the user message we just added so it doesn't get resent on next rerun,
            # then rerun once to show the error banner above the chat input.
            st.rerun()

# ---------------- Maintenance tab ----------------
with tab_maintenance:
    st.subheader(T["maintenance_header"])
    st.write(T["maintenance_desc"])
    if st.button(T["maintenance_button"], type="primary"):
        prompt = (
            f"{vehicle_context_str()}\n\n"
            "Generate a preventive maintenance checklist for this vehicle as described in your instructions."
        )
        reply = call_gemini(
            [types.Part.from_text(text=prompt)],
            BASE_MAINTENANCE_PROMPT,
            max_tokens=8000,
        )
        if reply:
            st.markdown(reply)
        elif st.session_state.last_error:
            st.error(st.session_state.last_error)

# ---------------- About tab ----------------
with tab_about:
    st.subheader(T["about_what_header"])
    st.markdown(T["about_what_body"])
    st.subheader(T["about_disclaimer_header"])
    st.warning(T["about_disclaimer_body"])
    st.caption(
        f"{T['session_started']}: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
