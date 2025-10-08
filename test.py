import streamlit as st
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# DATA MODELS AND DATABASE
# ============================================================================

@dataclass
class Phone:
    id: str
    brand: str
    model: str
    price: int
    camera_mp: int
    has_ois: bool
    battery_mah: int
    fast_charging_w: int
    screen_size: float
    processor: str
    ram_gb: int
    storage_gb: int
    os: str
    weight_g: int
    special_features: List[str]
    
    def to_dict(self):
        return {
            'id': self.id,
            'brand': self.brand,
            'model': self.model,
            'price': self.price,
            'camera_mp': self.camera_mp,
            'has_ois': self.has_ois,
            'battery_mah': self.battery_mah,
            'fast_charging_w': self.fast_charging_w,
            'screen_size': self.screen_size,
            'processor': self.processor,
            'ram_gb': self.ram_gb,
            'storage_gb': self.storage_gb,
            'os': self.os,
            'weight_g': self.weight_g,
            'special_features': self.special_features
        }

class QueryIntent(Enum):
    SEARCH = "search"
    COMPARE = "compare"
    EXPLAIN = "explain"
    DETAILS = "details"
    ADVERSARIAL = "adversarial"
    IRRELEVANT = "irrelevant"

# Mock Phone Database
PHONE_DATABASE = [
    Phone("p1", "Samsung", "Galaxy S23 FE", 29999, 50, True, 4500, 25, 6.4, "Exynos 2200", 8, 128, "Android", 209, ["IP68", "Wireless Charging"]),
    Phone("p2", "Google", "Pixel 8a", 52999, 64, True, 4492, 18, 6.1, "Tensor G3", 8, 128, "Android", 188, ["AI Camera", "7 Years Updates"]),
    Phone("p3", "OnePlus", "12R", 39999, 50, True, 5500, 100, 6.78, "Snapdragon 8 Gen 2", 8, 128, "Android", 207, ["120Hz AMOLED", "Fast Charging"]),
    Phone("p4", "Xiaomi", "Redmi Note 13 Pro+", 31999, 200, True, 5000, 120, 6.67, "Dimensity 7200", 8, 256, "Android", 199, ["200MP Camera", "Ultra Fast Charging"]),
    Phone("p5", "Samsung", "Galaxy A54", 34999, 50, True, 5000, 25, 6.4, "Exynos 1380", 8, 128, "Android", 202, ["IP67", "Super AMOLED"]),
    Phone("p6", "Realme", "11 Pro+", 27999, 200, True, 5000, 100, 6.7, "Dimensity 7050", 8, 256, "Android", 183, ["200MP Camera", "Curved Display"]),
    Phone("p7", "Motorola", "Edge 40", 24999, 50, True, 4400, 68, 6.55, "Dimensity 8020", 8, 256, "Android", 167, ["Compact Design", "IP68"]),
    Phone("p8", "Nothing", "Phone 2a", 23999, 50, False, 5000, 45, 6.7, "Dimensity 7200 Pro", 8, 128, "Android", 190, ["Glyph Interface", "Clean UI"]),
    Phone("p9", "iQOO", "Neo 7 Pro", 34999, 50, True, 5000, 120, 6.78, "Snapdragon 8+ Gen 1", 8, 128, "Android", 197, ["Gaming Phone", "Ultra Fast Charging"]),
    Phone("p10", "Poco", "X6 Pro", 26999, 64, True, 5000, 67, 6.67, "Dimensity 8300 Ultra", 8, 256, "Android", 186, ["Performance Beast", "Flow AMOLED"]),
    Phone("p11", "Vivo", "V29", 32999, 50, True, 4600, 80, 6.78, "Snapdragon 778G", 8, 128, "Android", 186, ["Aura Light", "Curved Display"]),
    Phone("p12", "Oppo", "Reno 11", 29999, 50, False, 5000, 67, 6.7, "Dimensity 7050", 8, 128, "Android", 184, ["Portrait Master", "3D Curved Screen"]),
    Phone("p13", "Samsung", "Galaxy M34", 16999, 50, True, 6000, 25, 6.5, "Exynos 1280", 6, 128, "Android", 208, ["Monster Battery", "Super AMOLED"]),
    Phone("p14", "Realme", "Narzo 60 Pro", 23999, 100, True, 5000, 67, 6.7, "Dimensity 7050", 8, 128, "Android", 191, ["100MP Camera", "Mars Orange Design"]),
    Phone("p15", "Xiaomi", "13T", 39999, 50, True, 5000, 67, 6.67, "Dimensity 8200 Ultra", 8, 256, "Android", 197, ["Leica Camera", "144Hz Display"]),
    # Budget phones under 15k
    Phone("p16", "Redmi", "13C 5G", 9999, 50, False, 5000, 18, 6.74, "Dimensity 6100+", 4, 128, "Android", 192, ["90Hz Display", "5G Ready"]),
    Phone("p17", "Realme", "Narzo N53", 8999, 50, False, 5000, 33, 6.74, "Unisoc T612", 4, 64, "Android", 182, ["33W Charging", "Slim Design"]),
    Phone("p18", "Poco", "M6 5G", 9999, 50, False, 5000, 18, 6.74, "Dimensity 6100+", 4, 128, "Android", 195, ["5G Support", "90Hz Display"]),
    Phone("p19", "Samsung", "Galaxy M14", 12990, 50, False, 6000, 25, 6.6, "Exynos 1330", 4, 128, "Android", 206, ["Massive Battery", "Super AMOLED"]),
    Phone("p20", "Motorola", "G54 5G", 13999, 50, False, 5000, 33, 6.5, "Dimensity 7020", 6, 128, "Android", 180, ["5G Support", "120Hz Display"]),
    Phone("p21", "Realme", "C55", 10999, 64, False, 5000, 33, 6.72, "Helio G88", 6, 128, "Android", 189, ["64MP Camera", "Premium Design"]),
    Phone("p22", "Redmi", "12 5G", 11999, 50, False, 5000, 22, 6.79, "Snapdragon 4 Gen 2", 6, 128, "Android", 199, ["5G Ready", "90Hz Display"]),
    Phone("p23", "Poco", "C65", 8499, 50, False, 5000, 18, 6.74, "Helio G85", 6, 128, "Android", 192, ["Big Battery", "Budget King"]),
    Phone("p24", "Lava", "Blaze 2 5G", 9999, 50, False, 5000, 18, 6.5, "Dimensity 6020", 4, 128, "Android", 185, ["Clean Android", "5G Support"]),
    Phone("p25", "Infinix", "Note 30 5G", 12999, 108, False, 5000, 45, 6.78, "Dimensity 6080", 8, 128, "Android", 195, ["108MP Camera", "JBL Speakers"]),
]

# ============================================================================
# ADVERSARIAL DETECTION & SAFETY
# ============================================================================

ADVERSARIAL_PATTERNS = [
    r"ignore.*(?:previous|above|prior|instructions|rules)",
    r"reveal.*(?:prompt|system|instructions|rules|api|key|secret)",
    r"show.*(?:prompt|system|instructions|rules|api|key|secret)",
    r"what.*(?:are|is).*your.*(?:prompt|instructions|rules|system)",
    r"tell me.*(?:prompt|instructions|rules|system|api|key)",
    r"bypass.*(?:security|rules|filters)",
    r"act as.*(?:developer|admin|root)",
    r"you are now",
    r"pretend.*(?:you're|you are)",
    r"roleplay",
    r"jailbreak",
    r"DAN mode",
]

TOXIC_PATTERNS = [
    r"trash.*(?:brand|phone|company)",
    r"garbage.*(?:brand|phone|company)",
    r"worst.*(?:brand|phone|company).*(?:ever|shit|crap)",
    r"(?:hate|destroy|kill).*(?:brand|company)",
]

IRRELEVANT_KEYWORDS = [
    "weather", "recipe", "joke", "story", "poem", "song", 
    "movie", "politics", "religion", "stock", "crypto"
]

def detect_adversarial(query: str) -> Tuple[bool, str]:
    """Detect adversarial or malicious queries."""
    query_lower = query.lower()
    
    # Check for adversarial patterns
    for pattern in ADVERSARIAL_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return True, "security"
    
    # Check for toxic patterns
    for pattern in TOXIC_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return True, "toxic"
    
    return False, ""

def detect_irrelevant(query: str) -> bool:
    """Detect queries unrelated to mobile phones."""
    query_lower = query.lower()
    
    # Check if query is about phones at all
    phone_keywords = ["phone", "mobile", "smartphone", "android", "ios", "iphone", 
                      "camera", "battery", "processor", "display", "screen"]
    
    has_phone_context = any(kw in query_lower for kw in phone_keywords)
    has_irrelevant = any(kw in query_lower for kw in IRRELEVANT_KEYWORDS)
    
    # If no phone context and has irrelevant keywords, mark as irrelevant
    if has_irrelevant and not has_phone_context:
        return True
    
    # If query is very short and has no phone keywords, it might be irrelevant
    if len(query.split()) < 3 and not has_phone_context:
        return True
    
    return False

# ============================================================================
# QUERY PARSING & INTENT DETECTION
# ============================================================================

def parse_query(query: str) -> Dict:
    """Parse user query to extract intent and parameters."""
    query_lower = query.lower()
    
    # Check for adversarial/toxic content first
    is_adversarial, adv_type = detect_adversarial(query)
    if is_adversarial:
        return {
            'intent': QueryIntent.ADVERSARIAL,
            'adversarial_type': adv_type,
            'original_query': query
        }
    
    # Check for irrelevant queries
    if detect_irrelevant(query):
        return {
            'intent': QueryIntent.IRRELEVANT,
            'original_query': query
        }
    
    # Extract budget
    budget = None
    budget_patterns = [
        r'under\s*₹?\s*(\d+)k',
        r'below\s*₹?\s*(\d+)k',
        r'around\s*₹?\s*(\d+)k',
        r'₹?\s*(\d+)k\s*budget',
        r'up\s*to\s*₹?\s*(\d+)k',
        r'under\s*₹?\s*(\d+)',
        r'below\s*₹?\s*(\d+)',
        r'around\s*₹?\s*(\d+)',
    ]
    for pattern in budget_patterns:
        match = re.search(pattern, query_lower)
        if match:
            amount = int(match.group(1))
            # If it's written as "15k", multiply by 1000
            if 'k' in match.group(0):
                budget = amount * 1000
            # If it's a number less than 1000, assume it's in thousands (e.g., "15" means 15000)
            elif amount < 1000:
                budget = amount * 1000
            else:
                budget = amount
            break
    
    # Extract brand preference
    brands = ["samsung", "google", "oneplus", "xiaomi", "realme", "motorola", 
              "nothing", "iqoo", "poco", "vivo", "oppo", "pixel"]
    brand = None
    for b in brands:
        if b in query_lower:
            brand = b.title()
            if b == "pixel":
                brand = "Google"
            break
    
    # Detect comparison intent
    if "compare" in query_lower or "vs" in query_lower or " versus " in query_lower:
        # Extract phone models
        models = []
        for phone in PHONE_DATABASE:
            full_name = f"{phone.brand} {phone.model}".lower()
            if full_name in query_lower or phone.model.lower() in query_lower:
                models.append(phone.id)
        
        return {
            'intent': QueryIntent.COMPARE,
            'phone_ids': models[:3],  # Max 3 phones
            'original_query': query
        }
    
    # Detect explanation intent
    explain_keywords = ["explain", "what is", "difference between", "tell me about"]
    if any(kw in query_lower for kw in explain_keywords):
        return {
            'intent': QueryIntent.EXPLAIN,
            'topic': query,
            'original_query': query
        }
    
    # Detect details intent
    if "tell me more" in query_lower or "details about" in query_lower or "this phone" in query_lower:
        return {
            'intent': QueryIntent.DETAILS,
            'original_query': query
        }
    
    # Extract feature preferences
    features = {
        'camera': any(kw in query_lower for kw in ["camera", "photo", "photography", "video"]),
        'battery': any(kw in query_lower for kw in ["battery", "charging", "power"]),
        'performance': any(kw in query_lower for kw in ["gaming", "performance", "fast", "processor"]),
        'compact': any(kw in query_lower for kw in ["compact", "small", "one hand", "light"]),
        'display': any(kw in query_lower for kw in ["display", "screen", "amoled"]),
    }
    
    return {
        'intent': QueryIntent.SEARCH,
        'budget': budget,
        'brand': brand,
        'features': features,
        'original_query': query
    }

# ============================================================================
# PHONE SEARCH & FILTERING
# ============================================================================

def search_phones(parsed_query: Dict) -> List[Phone]:
    """Search and filter phones based on parsed query."""
    results = PHONE_DATABASE.copy()
    
    # Filter by budget (add 10% buffer for "around" queries)
    if parsed_query.get('budget'):
        budget = parsed_query['budget']
        # Add buffer if query uses "around"
        if 'around' in parsed_query.get('original_query', '').lower():
            budget_max = budget * 1.15  # 15% upper buffer
            budget_min = budget * 0.85  # 15% lower buffer
            results = [p for p in results if budget_min <= p.price <= budget_max]
        else:
            results = [p for p in results if p.price <= budget]
    
    # Filter by brand
    if parsed_query.get('brand'):
        brand_lower = parsed_query['brand'].lower()
        results = [p for p in results if p.brand.lower() == brand_lower or 
                   (brand_lower == 'google' and p.brand.lower() == 'pixel') or
                   (brand_lower == 'redmi' and p.brand.lower() in ['redmi', 'xiaomi']) or
                   (brand_lower == 'xiaomi' and p.brand.lower() in ['redmi', 'xiaomi'])]
    
    # Score phones based on features
    scored_phones = []
    for phone in results:
        score = 0
        features = parsed_query.get('features', {})
        
        if features.get('camera'):
            score += phone.camera_mp / 10
            if phone.has_ois:
                score += 5
        
        if features.get('battery'):
            score += phone.battery_mah / 500
            score += phone.fast_charging_w / 10
        
        if features.get('performance'):
            score += phone.ram_gb * 2
            if "snapdragon 8" in phone.processor.lower() or "dimensity 8" in phone.processor.lower():
                score += 10
        
        if features.get('compact'):
            if phone.screen_size < 6.6:
                score += 10
            if phone.weight_g < 190:
                score += 5
        
        if features.get('display'):
            if "amoled" in str(phone.special_features).lower():
                score += 5
        
        scored_phones.append((phone, score))
    
    # Sort by score, then by price (lower is better)
    scored_phones.sort(key=lambda x: (-x[1], x[0].price))
    
    return [p[0] for p in scored_phones[:5]]

def get_phone_by_id(phone_id: str) -> Optional[Phone]:
    """Get phone by ID."""
    for phone in PHONE_DATABASE:
        if phone.id == phone_id:
            return phone
    return None

# ============================================================================
# RESPONSE GENERATION
# ============================================================================

def generate_comparison_response(phone_ids: List[str]) -> str:
    """Generate comparison response for phones."""
    phones = [get_phone_by_id(pid) for pid in phone_ids if get_phone_by_id(pid)]
    
    if len(phones) < 2:
        return "I need at least 2 phones to compare. Please specify the models you'd like to compare."
    
    response = f"## Comparing {len(phones)} Phones\n\n"
    
    # Price comparison
    response += "### 💰 Price\n"
    for phone in phones:
        response += f"- **{phone.brand} {phone.model}**: ₹{phone.price:,}\n"
    
    # Camera comparison
    response += "\n### 📸 Camera\n"
    for phone in phones:
        ois = "Yes ✓" if phone.has_ois else "No ✗"
        response += f"- **{phone.brand} {phone.model}**: {phone.camera_mp}MP, OIS: {ois}\n"
    
    # Battery comparison
    response += "\n### 🔋 Battery & Charging\n"
    for phone in phones:
        response += f"- **{phone.brand} {phone.model}**: {phone.battery_mah}mAh, {phone.fast_charging_w}W fast charging\n"
    
    # Performance comparison
    response += "\n### ⚡ Performance\n"
    for phone in phones:
        response += f"- **{phone.brand} {phone.model}**: {phone.processor}, {phone.ram_gb}GB RAM\n"
    
    # Trade-offs
    response += "\n### 🎯 Key Trade-offs\n\n"
    
    best_camera = max(phones, key=lambda p: p.camera_mp)
    response += f"**Best Camera**: {best_camera.brand} {best_camera.model} ({best_camera.camera_mp}MP)\n\n"
    
    best_battery = max(phones, key=lambda p: p.battery_mah)
    response += f"**Best Battery**: {best_battery.brand} {best_battery.model} ({best_battery.battery_mah}mAh)\n\n"
    
    fastest_charging = max(phones, key=lambda p: p.fast_charging_w)
    response += f"**Fastest Charging**: {fastest_charging.brand} {fastest_charging.model} ({fastest_charging.fast_charging_w}W)\n\n"
    
    cheapest = min(phones, key=lambda p: p.price)
    response += f"**Best Value**: {cheapest.brand} {cheapest.model} (₹{cheapest.price:,})\n\n"
    
    return response

def generate_explanation_response(topic: str) -> str:
    """Generate explanation for technical terms."""
    topic_lower = topic.lower()
    
    if "ois" in topic_lower and "eis" in topic_lower:
        return """## OIS vs EIS: Camera Stabilization Explained

**OIS (Optical Image Stabilization)**
- Hardware-based stabilization using physical lens movement
- Compensates for hand shake in real-time
- Better for low-light photography and video
- More expensive to implement
- Example: Moving lens elements counteract movement

**EIS (Electronic Image Stabilization)**
- Software-based stabilization using digital cropping
- Crops and shifts the frame to smooth out movement
- Can introduce slight quality loss due to cropping
- Cheaper to implement
- Works well in good lighting conditions

**Which is better?** OIS is generally superior, especially for photos and low-light scenarios. Many flagship phones use both OIS + EIS for best results.
"""
    
    elif "processor" in topic_lower or "chipset" in topic_lower:
        return """## Mobile Processors Explained

**Snapdragon (Qualcomm)**
- Most popular in premium Android phones
- Gen 8 series: Flagship performance
- Gen 7 series: Mid-range performance
- Excellent gaming and AI capabilities

**Dimensity (MediaTek)**
- Strong mid-range and upper mid-range options
- 7000-9000 series offer great value
- Improving gaming performance
- Power efficient

**Exynos (Samsung)**
- Used in Samsung phones
- Mixed reputation for efficiency
- Newer generations improving

**Tensor (Google)**
- Custom chip by Google for Pixel phones
- Optimized for AI and machine learning
- Excellent camera processing
"""
    
    elif "amoled" in topic_lower or "display" in topic_lower:
        return """## Display Technology Explained

**AMOLED (Active Matrix OLED)**
- Self-lit pixels (no backlight needed)
- True blacks and infinite contrast
- Vibrant colors and wide viewing angles
- More power efficient with dark themes
- Used in most premium phones

**Super AMOLED (Samsung)**
- Samsung's enhanced AMOLED
- Better sunlight visibility
- Touch sensors integrated into display

**LCD/IPS**
- Requires backlight
- Generally cheaper
- Can't achieve true blacks
- Still good color accuracy

**Refresh Rate**
- 60Hz: Standard
- 90Hz: Smoother scrolling
- 120Hz: Premium smoothness for gaming and UI
"""
    
    else:
        return "I can explain various phone-related terms like OIS vs EIS, processor types, display technology, camera specs, and more. What specific topic would you like me to explain?"

def generate_search_response(phones: List[Phone], parsed_query: Dict) -> str:
    """Generate search response with recommendations."""
    if not phones:
        return "Sorry, I couldn't find any phones matching your criteria. Try adjusting your budget or requirements."
    
    budget_str = f"under ₹{parsed_query.get('budget'):,}" if parsed_query.get('budget') else "in your range"
    brand_str = f" from {parsed_query.get('brand')}" if parsed_query.get('brand') else ""
    
    response = f"## 📱 Found {len(phones)} Great Options {budget_str}{brand_str}\n\n"
    
    # Top recommendation
    top_phone = phones[0]
    response += f"### 🏆 Top Recommendation: {top_phone.brand} {top_phone.model}\n"
    response += f"**Price**: ₹{top_phone.price:,}\n\n"
    
    # Explain why
    response += "**Why this phone?**\n"
    features = parsed_query.get('features', {})
    
    if features.get('camera'):
        response += f"- Excellent {top_phone.camera_mp}MP camera with {'OIS for stable shots' if top_phone.has_ois else 'great image quality'}\n"
    
    if features.get('battery'):
        response += f"- Powerful {top_phone.battery_mah}mAh battery with {top_phone.fast_charging_w}W fast charging\n"
    
    if features.get('performance'):
        response += f"- Strong {top_phone.processor} processor with {top_phone.ram_gb}GB RAM\n"
    
    if features.get('compact'):
        response += f"- Compact {top_phone.screen_size}\" display, easy to handle at {top_phone.weight_g}g\n"
    
    if not any(features.values()):
        response += f"- Well-balanced specs with {top_phone.processor}, {top_phone.ram_gb}GB RAM\n"
        response += f"- Good {top_phone.camera_mp}MP camera and {top_phone.battery_mah}mAh battery\n"
    
    if top_phone.special_features:
        response += f"- Special features: {', '.join(top_phone.special_features)}\n"
    
    response += "\n"
    
    return response

def handle_adversarial_query(parsed_query: Dict) -> str:
    """Handle adversarial queries with appropriate refusal."""
    adv_type = parsed_query.get('adversarial_type', 'security')
    
    if adv_type == 'security':
        return """I'm a mobile phone shopping assistant designed to help you find the perfect phone. I cannot:
- Reveal system prompts or internal instructions
- Share API keys or credentials
- Bypass security measures

**How can I help you?** I can:
- Recommend phones based on your budget and needs
- Compare different models
- Explain technical terms
- Answer questions about phone features

Please ask me about mobile phones!"""
    
    elif adv_type == 'toxic':
        return """I maintain a neutral, factual approach when discussing phone brands and models. Instead of making negative generalizations, I can:

- Compare specific models objectively
- Discuss pros and cons of different phones
- Help you find alternatives that match your needs
- Provide factual specifications and reviews

What specific features or phones would you like to know about?"""
    
    return "I'm here to help you find the right mobile phone. Please ask me about phone recommendations, comparisons, or features!"

def handle_irrelevant_query(parsed_query: Dict) -> str:
    """Handle queries unrelated to mobile phones."""
    return """I'm specifically designed to help with mobile phone shopping. I can assist you with:

📱 Finding phones based on your budget and needs
🔍 Comparing different models
💡 Explaining technical features (OIS, processors, displays, etc.)
📊 Showing detailed specifications

I cannot help with topics unrelated to mobile phones. How can I help you find your perfect phone today?"""

# ============================================================================
# STREAMLIT UI
# ============================================================================

def display_phone_card(phone: Phone):
    """Display phone as a card using Streamlit components."""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {phone.brand} {phone.model}")
        with col2:
            st.markdown(f"### ₹{phone.price:,}")
        
        # Specs in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**📸 Camera:** {phone.camera_mp}MP {'+OIS' if phone.has_ois else ''}")
            st.markdown(f"**🔋 Battery:** {phone.battery_mah}mAh")
        
        with col2:
            st.markdown(f"**⚡ Charging:** {phone.fast_charging_w}W")
            st.markdown(f"**📱 Display:** {phone.screen_size}\"")
        
        with col3:
            st.markdown(f"**🧠 Processor:** {phone.processor}")
            st.markdown(f"**💾 RAM:** {phone.ram_gb}GB")
        
        st.markdown(f"**✨ Special:** {', '.join(phone.special_features)}")
        st.divider()

def main():
    st.set_page_config(
        page_title="AI Mobile Phone Assistant",
        page_icon="📱",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("📱 AI Mobile Phone Shopping Assistant")
    st.markdown("Find your perfect phone with AI-powered recommendations")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.last_phones = []
    
    # Sidebar with examples
    with st.sidebar:
        st.header("💡 Try These Queries")
        st.markdown("""
        **Search Examples:**
        - Best camera phone under ₹30,000?
        - Compact Android with good one-hand use
        - Battery king around ₹15k
        - Show me Samsung phones under ₹25k
        
        **Compare:**
        - Compare Pixel 8a vs OnePlus 12R
        - Compare Galaxy S23 FE, Pixel 8a, and OnePlus 12R
        
        **Learn:**
        - Explain OIS vs EIS
        - What is AMOLED display?
        - Tell me about processors
        
        **Details:**
        - Tell me more about this phone
        (after seeing recommendations)
        """)
        
        st.markdown("---")
        st.markdown("### 🛡️ Safety Features")
        st.markdown("""
        - Adversarial prompt detection
        - Toxic content filtering
        - Irrelevant query handling
        - Factual, neutral responses
        """)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.last_phones = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display phone cards if present
            if "phones" in message and message["phones"]:
                for phone in message["phones"]:
                    display_phone_card(phone)
    
    # Chat input
    if prompt := st.chat_input("Ask me about mobile phones..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Parse query
                parsed_query = parse_query(prompt)
                
                # Handle different intents
                if parsed_query['intent'] == QueryIntent.ADVERSARIAL:
                    response = handle_adversarial_query(parsed_query)
                    phones = []
                
                elif parsed_query['intent'] == QueryIntent.IRRELEVANT:
                    response = handle_irrelevant_query(parsed_query)
                    phones = []
                
                elif parsed_query['intent'] == QueryIntent.COMPARE:
                    response = generate_comparison_response(parsed_query.get('phone_ids', []))
                    phones = []
                
                elif parsed_query['intent'] == QueryIntent.EXPLAIN:
                    response = generate_explanation_response(parsed_query.get('topic', ''))
                    phones = []
                
                elif parsed_query['intent'] == QueryIntent.DETAILS:
                    if st.session_state.last_phones:
                        phone = st.session_state.last_phones[0]
                        response = f"## {phone.brand} {phone.model} - Detailed Specs\n\n"
                        response += f"**Price**: ₹{phone.price:,}\n\n"
                        response += f"**Camera**: {phone.camera_mp}MP with {'OIS' if phone.has_ois else 'EIS'}\n\n"
                        response += f"**Battery**: {phone.battery_mah}mAh with {phone.fast_charging_w}W fast charging\n\n"
                        response += f"**Display**: {phone.screen_size}\" screen\n\n"
                        response += f"**Processor**: {phone.processor}\n\n"
                        response += f"**Memory**: {phone.ram_gb}GB RAM + {phone.storage_gb}GB Storage\n\n"
                        response += f"**Weight**: {phone.weight_g}g\n\n"
                        response += f"**Special Features**: {', '.join(phone.special_features)}\n\n"
                        phones = [phone]
                    else:
                        response = "Please search for phones first, then I can provide detailed information."
                        phones = []
                
                else:  # SEARCH
                    phones = search_phones(parsed_query)
                    response = generate_search_response(phones, parsed_query)
                    st.session_state.last_phones = phones
                
                # Display response
                st.markdown(response)
                
                # Display phone cards for search results
                if phones:
                    for phone in phones:
                        display_phone_card(phone)
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "phones": phones if phones else []
        })

if __name__ == "__main__":
    main()
