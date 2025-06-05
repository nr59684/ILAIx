from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import nltk

# --- Load your saved model, tokenizer, and mlb ---
model_dir = "../../model/Bert2.0"  # Replace with your model directory
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

loaded_data = np.load("../mlb.npz", allow_pickle=True)  # Replace with your path
loaded_classes = loaded_data["classes"]
mlb = MultiLabelBinarizer()
mlb.classes_ = loaded_classes

# --- Sentence Segmentation ---
def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# --- Custom Predictor for Sentence-Based LIME ---
def sentence_predictor_for_lime(binary_vectors, original_sentences):
    texts = []
    for binary_vector in binary_vectors:
        selected_sentences = [s for i, s in enumerate(original_sentences) if binary_vector[i] == 1]
        text = " ".join(selected_sentences)
        texts.append(text)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()

    return probs

# --- LIME Explainer ---
class_names = mlb.classes_.tolist()
explainer = LimeTextExplainer(class_names=class_names)

# --- Modified explain_with_lime function ---
def explain_with_lime_sentences(text, labels_to_explain, num_samples=1000):
    sentences = split_into_sentences(text)
    num_sentences = len(sentences)

    def predictor_fn(binary_vectors):
        return sentence_predictor_for_lime(binary_vectors, sentences)

    explanations = {}
    for label_index in labels_to_explain:
        label_name = class_names[label_index]

        # Generate perturbations: vectors of 0s and 1s
        data = np.random.randint(0, 2, size=(num_samples, num_sentences))
        
        # Get predictions for the perturbations
        predictions = predictor_fn(data)

        # Create a dummy explanation object
        exp = lime.explanation.Explanation(
            domain_mapper=None,  # We'll set domain_mapper later
            mode="classification",
            class_names=class_names
        )
        exp.predict_proba = predictions[0] # Set predict_proba
        
        # Fit a linear model to these predictions
        # (This is a simplified version of what LIME does internally)
        distance = np.sqrt(((data - np.ones(num_sentences))**2).sum(axis=1)) # Example distance metric
        weights = np.exp(-distance**2 / (0.75 * num_sentences)) # Example kernel function
        
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1) # Example linear model
        model.fit(data, predictions[:, label_index], sample_weight=weights)
        
        # Create the explanation
        explanation = []
        for i, coef in enumerate(model.coef_):
            explanation.append((sentences[i], coef))

        # Sort the explanation by the absolute value of the coefficients
        explanation.sort(key=lambda x: abs(x[1]), reverse=True)

        explanations[label_name] = explanation

    return explanations

# --- Function to explain a single text instance ---
def explain_text(text, num_features=10, num_samples=2000):
    """
    Explains a single text instance using LIME with sentence-level perturbations.

    Args:
        text: The input text (string).
        num_features: The number of features (sentences) to include in the explanation.
        num_samples: The number of perturbed samples to generate.

    Returns:
        A dictionary where keys are label names and values are lists of
        (sentence, weight) tuples representing the explanation.
    """

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    predicted_label_indices = [i for i, prob in enumerate(predicted_probs) if prob > 0.5]
    predicted_labels_one_hot = np.zeros((1, len(mlb.classes_)))
    predicted_labels_one_hot[0, predicted_label_indices] = 1
    predicted_label_names = mlb.inverse_transform(predicted_labels_one_hot)[0]

    print("-" * 50)
    print(f"Input Text:\n{text}\n")
    print(f"Predicted Labels: {predicted_label_names}")

    explanations = explain_with_lime_sentences(text, predicted_label_indices, num_samples=num_samples)

    for label_name, sentences_weights in explanations.items():
        print(f"\nExplanation for label '{label_name}':")
        if sentences_weights:
            print("Sentences and Weights:")
            sorted_sentences_weights = sorted(sentences_weights, key=lambda x: abs(x[1]), reverse=True)
            for sentence, weight in sorted_sentences_weights:
                print(f"- {sentence} (weight: {weight:.3f})")
        else:
            print("No significant sentences found for this label.")

    return explanations

# --- Main Function ---
if __name__ == "__main__":
    license_text="This document may be copied, in whole or in part, in any form or by any means, as is or with alterations, provided that (1) alterations are clearly marked as alterations and (2) this copyright notice is included unmodified in any copy."

    explain_text(license_text)