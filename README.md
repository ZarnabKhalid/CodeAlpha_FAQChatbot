# CodeAlpha_FAQChatbot

An FAQ chatbot for **Khalid Shopping Centre**, a general store selling bags, travelling bags, makeup products and daily-use items (shampoo, lotion, etc.).

Built as **Task 2: Chatbot for FAQs** of the CodeAlpha Artificial Intelligence Internship.

## How it works

1. **FAQ data:** Store questions and answers (timings, location, delivery, payment, returns, products, contact). Each answer has several question variants in English and Roman Urdu.
2. **Preprocessing (NLTK):** lowercase, remove punctuation, tokenize, remove stopwords, lemmatize.
3. **Matching:** Questions are converted to TF-IDF vectors and the user's question is compared with every FAQ using **cosine similarity**.
4. **Response:** The answer of the most similar FAQ is shown. If the similarity is too low, a fallback message with the store's contact number is shown.
5. **Chat UI (optional):** A simple Tkinter chat window.

## Tech stack

- Python 3
- NLTK
- scikit-learn (TF-IDF, cosine similarity)
- Tkinter (GUI)

## Installation

```bash
pip install nltk scikit-learn
```

## Usage

Terminal chat:

```bash
python faq_chatbot.py
```

Chat window:

```bash
python faq_chatbot.py --gui
```

## Example

```
You: what time do you close
Bot: We are open every day from 8:00 AM to 9:00 PM.

You: delivery hoti hai?
Bot: Sorry, we do not offer home delivery. Please visit the shop to buy.

You: can I pay by card
Bot: We accept cash only.
```

## Customizing

Edit the `FAQS` list and the `PHONE` value at the top of `faq_chatbot.py` to use the bot for any other store or topic.

## Author

Zarnab Khalid - CodeAlpha AI Intern
