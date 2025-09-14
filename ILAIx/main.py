from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, abort
)
import json
import os
import re
from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from functools import wraps
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np  # Import numpy
from markupsafe import escape
from nltk.tokenize import sent_tokenize
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from lime.lime_text import LimeTextExplainer

ADMIN_USER = "admin"
ADMIN_PASS = "secret123"

# --- Preprocessing Functions ---

def admin_required(view):
    """Decorator: protect view with session check."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if session.get("is_admin"):
            return view(*args, **kwargs)
        return redirect(url_for('admin_login', next=request.path))
    return wrapped

def remove_html_tags(text):
    """Removes HTML tags from text."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")

def clean_special_chars(text):
    """
    Removes/replaces special characters and URLs from text.
    """
    # Remove URLs
    text = re.sub(r"http\S+", "", text)  # Removes URLs starting with "http" or "https"

    # Remove characters that are not alphanumeric, whitespace, or basic punctuation
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", "", text)
    
    return cleaned_text

def normalize_whitespace(text):
    """Normalizes whitespace in text."""
    cleaned_text = " ".join(text.split())
    return cleaned_text.strip()

def preprocess_text(text):
    """Applies all preprocessing steps to text."""
    text = remove_html_tags(text)
    text = clean_special_chars(text)
    text = normalize_whitespace(text)
    text = text.lower()  # Lowercasing
    return text

CHATS_FILE = "chats.json"  # File to store chats

def load_chats():
    """Loads chats from the JSON file."""
    try:
        with open(CHATS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []  # Return empty list if file doesn't exist

def save_chats(chats):
    """Saves chats to the JSON file."""
    with open(CHATS_FILE, "w") as f:
        json.dump(chats, f, indent=4) # Indent for readability


def summarize_attributions(attributions):
    """
    Sums attributions to get a single score per token.
    """
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def predictor(texts):
    """
    Prediction function for LIME.
    """
    with torch.no_grad():
        encoded_input = tokenizer(texts, truncation=True, padding=True,
                                  return_tensors="pt")
        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.sigmoid(outputs.logits).cpu().numpy()
    return probabilities

def get_top_sentences_with_lime(text, preprocessed_text, predicted_labels, top_k=3):
    """
    Uses LIME to explain predictions and returns top sentences per label.
    """
    explainer = LimeTextExplainer(class_names=mlb.classes_)
    result_dict = {}

    for label_index, label in enumerate(predicted_labels[0]):
      exp = explainer.explain_instance(preprocessed_text,
                                      predictor,
                                      num_features=25,
                                      top_labels=num_labels,  # Explain all possible labels
                                      num_samples=100)

      # Get word importance for the *specific* label index
      word_importance = {word: weight for word, weight in exp.as_list(label=label_index + 1)} #since top_labels is more than 1

      sentences = sent_tokenize(text)
      preprocessed_sentences = [preprocess_text(s) for s in sentences]
      sentence_scores = []

      for sent_idx, sent in enumerate(preprocessed_sentences):
          tokens = tokenizer.tokenize(sent)
          score = sum(word_importance.get(token, 0.0) for token in tokens)
          sentence_scores.append((sentences[sent_idx], score))

      sentence_scores.sort(key=lambda x: x[1], reverse=True)
      top_sentences = [sentence for sentence, _ in sentence_scores[:top_k]]
      result_dict[label] = top_sentences
      print(result_dict)
    return result_dict

def _stringify_labels(answer_field):
    """
    Accepts:
      - ["Display notice", "No trademark use", ...]               list[str]
      - [["Copyleft", "Source code"], ["Display notice"]]         list[list[str]]
    Returns a flat list[str] suitable for '<br>'.join()
    """
    flat = []
    for item in answer_field:
        if isinstance(item, (list, tuple, set)):
            flat.extend(str(x) for x in item)
        else:
            flat.append(str(item))
    # remove accidental duplicates while preserving order
    return list(dict.fromkeys(flat))


app = Flask(__name__)
app.jinja_env.filters['escapejs'] = escape
loaded_data = np.load("../src/mlb.npz", allow_pickle=True)
loaded_classes = loaded_data["classes"]
mlb = MultiLabelBinarizer()
mlb.classes_ = loaded_classes
num_labels = len(mlb.classes_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=''
tokenizer=''
lig=''
predicted_labels=[]
input_ids=''
attention_mask=''
token_type_ids=''
baseline_ids=''
question=''
chats = load_chats()
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-only-secret")

def load_model(modelName):
    global model, tokenizer, lig
    modelName = modelName.replace('\u00A0', ' ')
    if modelName == "FOSS Bert 2.0":
        modelName = "LegalBert"
    elif modelName == "FOSS Bert 1.5":
        modelName = "Roberta"
    elif modelName == "FOSS Bert 1.0":
        modelName = "Bert2.0"
    else:
        print("Invalid model name:" + modelName + "please check again")
        return
    model_directory = f"../model/{modelName}"
    model = AutoModelForSequenceClassification.from_pretrained(model_directory)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model.to(device)
    lig = LayerIntegratedGradients(predictor, model.base_model.embeddings)

@app.route('/chat/<license_name>')
def goChat(license_name):
    global chats
    existing_chat = next((chat for chat in chats if chat["license_name"] == license_name), None)
    data={"mychats":chats, "license_name":license_name}
    if existing_chat:
        # Escape for Javascript (Client Side)
        question = existing_chat['question']
        answer = existing_chat['answer']
        model_name=existing_chat['model']
        answer_html = "<br>".join(answer[0])
        data["result"]={"question":question,"answer":answer_html, "model":model_name}
    else:
        data["result"]= {"answer": "Chat not found, 404"}
    return render_template("chat.html", result=data)  # New route

@app.route("/")
def home():
    return render_template("index.html", myChats=chats)

@app.route("/loadModel", methods=["GET", "POST"])
def loadModel():
    data = {}
    global model, tokenizer, lig
    if request.method == "POST":
        try:
            modelName = request.json.get("modelName")
            print(modelName)
            load_model(modelName)
            data = {"result": "model loaded succesfully"}
        except:
            return None
    return jsonify(data)

@app.route("/getExplanations", methods=["GET", "POST"])
def getExplanations():
    global chats
    if request.method == "POST":
        licenseName = request.json.get("licenseName")
        question = request.json.get("question")
        existing_chat = next((chat for chat in chats if chat["license_name"] == licenseName), chats[0])
        if existing_chat:
            question = existing_chat['question']
            modelName=existing_chat['model']
            if existing_chat['explanations']=={}:
                predicted_labels=[tuple(existing_chat['answer'][0])]
                load_model(modelName)
                predicted_labels=[tuple(existing_chat['answer'][0])]
                preprocessed_text = preprocess_text(question)
                print(predicted_labels)
                result = get_top_sentences_with_lime(question, preprocessed_text, predicted_labels, top_k=3)
                existing_chat['explanations']=result
                for chat in chats:
                    if chat['license_name']==licenseName:
                        chat['explanations']=result
                        break
                save_chats(chats)
            else:
                result=existing_chat['explanations']
        print(result)
    return jsonify(result)

@app.route('/annotations', methods=['GET'])
@admin_required
def annotations():
    """
    Render a table of every chat stored in chats.json so the user can
    pick one to annotate.  Each row links to /annotate/<license_name>.
    """
    global chats
    # Build a lightweight list for the template
    rows = [
        {
            "license_name": c["license_name"],
            "model":        c["model"],
            "num_labels":   len(_flatten_unique(c.get("answer", []))),
            "done":         c.get("user_edited", False)        # ✔ if already annotated
        }
        for c in chats
    ]
    return render_template("annotations.html", rows=rows)

def _flatten_unique(nested):
    """Turn list / list-of-lists into a de-duplicated flat list."""
    flat = []
    for item in nested:
        if isinstance(item, (list, tuple, set)):
            flat.extend(item)
        else:
            flat.append(item)
    # preserve order, drop duplicates
    return list(dict.fromkeys(flat))


@app.route('/annotate/<license_name>', methods=['GET', 'POST'])
@admin_required
def annotate_license(license_name: str):
    """
    GET  → show two-pane annotation UI for the selected license.
    POST → accept updated `explanations` JSON and persist it.
    """
    global chats,mlb
    chat = next((c for c in chats if c["license_name"] == license_name), None)

     # ── POST: user clicked “Save” in the UI ──────────────────────────────
    if request.method == 'POST':
        payload = request.get_json(force=True, silent=True) or {}
        chat["explanations"] = payload.get("explanations", {})
        chat["answer"]      = payload.get("labels", [])
        chat["user_edited"]  = True 
        save_chats(chats)
        return jsonify({"status": "ok"})
    
    raw_labels = chat.get("answer", [])              # list or list-of-lists
    predicted_labels = _flatten_unique(raw_labels) # ✓ predicted
    all_labels        = list(mlb.classes_)   # <-- new line
    ordered_labels = predicted_labels + [ lab for lab in all_labels if lab not in predicted_labels ]

    ...
    return render_template(
        "annotate.html",
        license_name      = chat["license_name"],
        full_text         = chat["question"],
        ordered_labels    = ordered_labels,
        predicted_labels  = predicted_labels,        # now a flat list[str]
        explanations      = chat.get("explanations", {})
    )

@app.route("/api", methods=["GET", "POST"])
def qa():
    global chats, input_ids, attention_mask, token_type_ids, baseline_ids, attributions, predicted_labels, question
    if request.method == "POST":
        question = request.json.get("question")
        modelName=request.json.get("modelName")
        name_index=question.index(":")
        license_name = question[:name_index].strip()
        question = question[name_index+1:]
        existing_chat = next((chat for chat in chats if chat["license_name"] == license_name), None)
        if existing_chat:
            response_html = "<br>".join(_stringify_labels(existing_chat["answer"]))
            data = {"answer": response_html}
            return jsonify(data)
        else:
            probabilities = predictor([question])  # Predict on preprocessed text
            predictions = (probabilities > 0.5).astype(int)
            response = mlb.inverse_transform(predictions)
            predicted_labels=response
            response_html = "<br>".join(_stringify_labels(response[0]))
            data = {"answer": response_html}
            chats.insert(0,{"license_name": license_name, "model": modelName, "question": question, "answer": response, "explanations": {}})
            save_chats(chats)
            return jsonify(data)
    data = {"result": "Thank you! I'm just a machine learning model designed to respond to questions and generate text based on my training data. Is there anything specific you'd like to ask or discuss? "}
        
    return jsonify(data)

# ── login/logout ─────────────────────────────────────────────────────────
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    err = None
    if request.method == 'POST':
        u = request.form.get('username', '')
        p = request.form.get('password', '')
        if u == ADMIN_USER and p == ADMIN_PASS:
            session['is_admin'] = True
            nxt = request.args.get('next') or url_for('annotations')
            return redirect(nxt)
        err = "Invalid credentials"
    return render_template('admin_login.html', error=err)

@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect(url_for('admin_login'))

if __name__ == "__main__":
    app.run(debug=False)