"""
Khalid Shopping Centre - FAQ Chatbot (CodeAlpha AI Internship, Task 2)
Steps: FAQs -> NLTK preprocessing -> TF-IDF + cosine similarity -> best answer
Install:  pip install nltk scikit-learn
Run:      python faq_chatbot.py          (terminal chat)
          python faq_chatbot.py --gui    (chat window)
"""
import re
import sys

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STORE = "Khalid Shopping Centre"
PHONE = "0300 6762016"  # <-- apna asli phone/WhatsApp number yahan likho

# ---------------------------------------------------------------------------
# 1. FAQ DATA: each answer has several ways of asking (English + Roman Urdu)
# ---------------------------------------------------------------------------
FAQS = [
    {
        "questions": ["hello", "hi", "hey there", "assalam o alaikum", "salam", "aoa"],
        "answer": f"Assalam o Alaikum! Welcome to {STORE}. Ask me about our timings, products, payment or location.",
    },
    {
        "questions": [
            "What are your shop timings?", "What time do you open?", "What time do you close?",
            "Is the shop open now?", "Opening hours",
            "Dukaan kitne baje khulti hai?", "Shop kab band hoti hai?", "Timing kya hai?",
        ],
        "answer": "We are open every day from 8:00 AM to 9:00 PM. (Hamari dukaan subah 8 baje se raat 9 baje tak khuli hoti hai.)",
    },
    {
        "questions": [
            "Where is your shop located?", "What is your address?", "How can I find the store?",
            "Where is Khalid Shopping Centre?", "Store location",
            "Shop kahan hai?", "Aap ka address kya hai?", "Dukaan ka pata batao",
        ],
        "answer": f"You can visit us at {STORE}. (Hum {STORE} mein milte hain.)",
    },
    {
        "questions": [
            "Do you offer home delivery?", "Can you deliver to my house?", "Do you deliver orders?",
            "Delivery available?", "Can I order online?",
            "Kya ghar tak delivery hoti hai?", "Delivery milti hai?", "Home delivery hai?",
        ],
        "answer": "Sorry, we do not offer home delivery. Please visit the shop to buy. (Maazrat, delivery ki sahulat nahi hai. Aap dukaan par aa kar kharid sakte hain.)",
    },
    {
        "questions": [
            "What payment methods do you accept?", "Can I pay by card?", "Do you accept Easypaisa or JazzCash?",
            "Is cash the only payment option?", "How can I pay?", "Do you take credit card?",
            "Payment kaise karni hai?", "Kya card chalta hai?", "Easypaisa JazzCash milta hai?",
        ],
        "answer": "We accept cash only. (Hum sirf cash payment lete hain.)",
    },
    {
        "questions": [
            "What is your return policy?", "Can I return an item?", "Can I exchange my product?",
            "Do you give a refund?", "Can I get my money back?", "Exchange policy",
            "Saman wapis ho sakta hai?", "Exchange hota hai?", "Paise wapis milte hain?", "Return policy kya hai?",
        ],
        "answer": "Sorry, we have a no return and no exchange policy. Please check items carefully before buying. (Saman wapis ya exchange nahi hota, is liye kharidne se pehle achi tarah check kar len.)",
    },
    {
        "questions": [
            "How can I contact you?", "What is your phone number?", "Do you have a WhatsApp number?",
            "How do I call the shop?", "Contact number",
            "Rabta kaise karun?", "Phone number kya hai?", "WhatsApp number batao",
        ],
        "answer": f"You can call or WhatsApp us on {PHONE}. (Aap is number par call ya WhatsApp kar sakte hain: {PHONE}.)",
    },
    {
        "questions": [
            "What do you sell?", "What products do you have?", "What items are available in your store?",
            "Do you have daily use items?", "Which things can I buy here?",
            "Aap ke paas kya kya milta hai?", "Kon kon si cheezein milti hain?", "Kya kya available hai?",
        ],
        "answer": "We sell bags, travelling bags, makeup products and daily-use items such as shampoo and lotion. (Hamare paas bags, travelling bags, makeup aur rozmarra ki cheezein jaise shampoo aur lotion milte hain.)",
    },
    {
        "questions": [
            "Do you sell bags?", "Do you have travelling bags?", "Do you have travel bags or suitcases?",
            "Handbags available?", "School bags?",
            "Bags milte hain?", "Travelling bag hai?", "Safar wale bag milte hain?",
        ],
        "answer": "Yes, we have bags and travelling bags. Please visit the shop to see the full collection. (Ji haan, bags aur travelling bags available hain. Poora collection dukaan par dekhen.)",
    },
    {
        "questions": [
            "Do you sell makeup?", "Do you have makeup products?", "Cosmetics available?",
            "Do you have lipstick or foundation?", "Beauty products",
            "Makeup ka saman milta hai?", "Cosmetics hain?", "Lipstick milti hai?",
        ],
        "answer": "Yes, we have a range of makeup and cosmetic products. (Ji haan, makeup aur cosmetics ka saman available hai.)",
    },
    {
        "questions": [
            "Do you sell shampoo?", "Do you have lotion?", "Do you have soap or face wash?",
            "Hair care and skin care products?", "Personal care items",
            "Shampoo milta hai?", "Lotion hai?", "Rozmarra ki cheezein milti hain?",
        ],
        "answer": "Yes, we have shampoo, lotion and other daily-use personal care products. (Ji haan, shampoo, lotion aur rozmarra ki cheezein available hain.)",
    },
    {
        "questions": [
            "Do you have any discount?", "Is there any sale going on?", "What are your prices?",
            "How much does it cost?", "Can I get a cheaper price?",
            "Koi discount hai?", "Sale chal rahi hai?", "Rate kya hai?", "Price kitni hai?",
        ],
        "answer": f"Prices and offers can be checked in the shop. For details, call {PHONE}. (Rate aur offers dukaan par dekhen ya {PHONE} par pooch len.)",
    },
]

FALLBACK = f"Sorry, I could not understand. Please call us on {PHONE} or visit the shop. (Maazrat, samajh nahi aaya. Aap {PHONE} par rabta karen.)"
THRESHOLD = 0.2  # below this similarity -> fallback

# ---------------------------------------------------------------------------
# 2. PREPROCESSING (NLTK)
# ---------------------------------------------------------------------------
for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

URDU_STOPWORDS = {"kya", "hai", "hain", "ka", "ki", "ke", "ko", "me", "mein", "se", "kar", "kr",
                  "ho", "hota", "hoti", "aap", "ap", "mjy", "mujhe", "kia", "to", "bhi", "na"}
STOP_WORDS = set(stopwords.words("english")) | URDU_STOPWORDS
LEMMATIZER = WordNetLemmatizer()


def preprocess(text):
    """lowercase -> remove punctuation -> tokenize -> remove stopwords -> lemmatize"""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 3. MATCHING (TF-IDF + cosine similarity)
# ---------------------------------------------------------------------------
class FAQBot:
    def __init__(self, faqs):
        self.questions, self.answers = [], []
        for item in faqs:
            for q in item["questions"]:
                self.questions.append(q)
                self.answers.append(item["answer"])
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.vectors = self.vectorizer.fit_transform([preprocess(q) for q in self.questions])

    def reply(self, user_input):
        cleaned = preprocess(user_input)
        if not cleaned.strip():
            return FALLBACK
        scores = cosine_similarity(self.vectorizer.transform([cleaned]), self.vectors)[0]
        best = scores.argmax()
        return self.answers[best] if scores[best] >= THRESHOLD else FALLBACK


# ---------------------------------------------------------------------------
# 4. INTERFACES
# ---------------------------------------------------------------------------
def run_cli(bot):
    print(f"{STORE} Bot: Assalam o Alaikum! Sawal poochen (type 'quit' to exit).")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit", "bye"}:
            print("Bot: Allah Hafiz!")
            break
        print("Bot:", bot.reply(user))


def run_gui(bot):
    import tkinter as tk
    from tkinter import scrolledtext

    root = tk.Tk()
    root.title(f"{STORE} - FAQ Chatbot")
    chat = scrolledtext.ScrolledText(root, width=65, height=20, state="disabled", wrap="word")
    chat.pack(padx=10, pady=10)
    entry = tk.Entry(root, width=55)
    entry.pack(side="left", padx=(10, 5), pady=(0, 10))

    def add(sender, msg):
        chat.config(state="normal")
        chat.insert("end", f"{sender}: {msg}\n\n")
        chat.config(state="disabled")
        chat.see("end")

    def send(event=None):
        text = entry.get().strip()
        if text:
            add("You", text)
            add("Bot", bot.reply(text))
            entry.delete(0, "end")

    entry.bind("<Return>", send)
    tk.Button(root, text="Send", command=send).pack(side="left", pady=(0, 10))
    add("Bot", f"Assalam o Alaikum! Welcome to {STORE}. Ask me anything.")
    root.mainloop()


if __name__ == "__main__":
    bot = FAQBot(FAQS)
    run_gui(bot) if "--gui" in sys.argv else run_cli(bot)
