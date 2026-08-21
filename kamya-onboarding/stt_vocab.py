"""Domain vocabulary for Deepgram Flux keyterm boosting.

A general multilingual model has never heard a Kamya consultation. It handles
"main subah kuch nahi khati" fine and then mangles "PCOS", "thyroid",
"methi", "besan". Those are the words the whole conversation turns on, so
Flux is told to weight them.

Ordered by cost of getting it wrong: a misheard CONDITION or MEDICATION can
change advice; a misheard food is merely annoying. Keep the list tight —
boosting everything boosts nothing.

Override wholesale with DEEPGRAM_KEYTERMS="term one,term two,...".
"""

# Conditions and clinical terms. Highest cost if misrecognised.
CONDITIONS = [
    "PCOS", "PCOD", "thyroid", "hypothyroid", "hyperthyroid", "diabetes",
    "prediabetes", "insulin resistance", "cholesterol", "triglycerides",
    "hypertension", "blood pressure", "anaemia", "anemia", "haemoglobin",
    "vitamin D", "vitamin B12", "fatty liver", "IBS", "acidity", "gastritis",
    "bloating", "constipation", "uric acid", "creatinine", "HbA1c",
    "fissure", "piles", "migraine", "endometriosis", "fibroid",
]

# Nutrition vocabulary Mira uses constantly.
NUTRITION = [
    "protein", "carbs", "carbohydrates", "fibre", "fiber", "calories",
    "calcium", "iron", "omega", "probiotic", "electrolyte", "portion",
    "metabolism", "deficiency", "supplement", "intermittent fasting",
]

# Indian foods. These are the ones a general model most reliably fumbles.
FOODS = [
    "dal", "roti", "chapati", "sabzi", "rajma", "chana", "chole", "paneer",
    "curd", "dahi", "chaas", "buttermilk", "ghee", "jaggery", "gud",
    "poha", "upma", "idli", "dosa", "sambhar", "khichdi", "daliya",
    "besan", "chilla", "methi", "palak", "lauki", "karela", "bhindi",
    "tinda", "arbi", "sooji", "suji", "atta", "maida", "bajra", "jowar",
    "ragi", "makhana", "chia", "sattu", "sprouts", "moong", "masoor",
    "toor", "urad", "paratha", "puri", "samosa", "pakora", "biryani",
    "rice", "chawal", "milk", "doodh", "chai", "coffee", "almonds",
    "badam", "walnut", "akhrot", "peanut", "moongfali", "til", "flaxseed",
    "alsi", "coconut", "nariyal", "banana", "kela", "papaya", "papita",
    "guava", "amrud", "apple", "seb", "orange", "santra",
]

# Meal-timing words that drive the six-slot eating pattern.
TIMING = [
    "breakfast", "lunch", "dinner", "nashta", "subah", "dopahar", "shaam",
    "raat", "snack", "mid morning", "evening snack", "late night",
    "khaali pet", "empty stomach", "before sleep",
]

DEFAULT_KEYTERMS = CONDITIONS + NUTRITION + FOODS + TIMING
