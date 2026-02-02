# Support Ticket Classification & Prioritization System

This project implements a real-world NLP-based decision-support system for automatically classifying and prioritizing customer support and IT service tickets. The system is designed to reduce manual triage effort, improve response times for critical issues, and ensure safe automation using confidence-aware routing.

---

## Problem Statement

Customer support teams receive a large volume of tickets daily. Manual categorization and prioritization lead to delays, inconsistent handling, and increased operational cost. This project addresses these challenges by building a machine learning system that:

- Automatically classifies support tickets into relevant categories
- Assigns priority levels (High / Medium / Low)
- Routes low-confidence cases for manual review
- Acts as a decision-support system rather than blind automation

---

## Solution Overview

The system uses a hybrid NLP and rule-based approach:

- **TF-IDF features** capture domain-specific keywords
- **Sentence embeddings (Sentence-BERT)** capture semantic meaning
- **Ensemble learning** combines both representations for robust predictions
- **Confidence-aware business rules** ensure safe routing and prioritization

---

## Dataset Overview

The model is trained on a real-world IT service ticket dataset containing textual descriptions and labeled ticket categories.

### 🔹 Screenshot: Dataset Preview
<img width="277" height="115" alt="image" src="https://github.com/user-attachments/assets/0a6db206-0e6b-46d5-a548-6298a7642416" />


---

## Exploratory Data Analysis (EDA)

EDA was performed to understand ticket distribution and text characteristics.

Key analyses include:
- Ticket category distribution
- Character length distribution
- Word count analysis

### Screenshot: Ticket Category Distribution
<img width="975" height="547" alt="image" src="https://github.com/user-attachments/assets/bc1e30a1-9edd-42c6-ae3b-89cd4415222f" />


### 🔹 Screenshot: Character / Word Length Distribution
<img width="558" height="393" alt="image" src="https://github.com/user-attachments/assets/b02aaa03-f8d1-4ccd-b7aa-97a1b0913627" />


---

## Model Development

### Text Preprocessing
- Lowercasing
- Removal of punctuation and special characters
- Whitespace normalization

### Feature Engineering
- TF-IDF with unigrams and bigrams
- Sentence embeddings using `all-MiniLM-L6-v2`

### Models Trained
- Multinomial Naive Bayes
- Logistic Regression
- SGD Classifier
- Linear SVM
- Decision Tree
- Random Forest
- KNN
- **Ensemble (TF-IDF + Embedding)**

---

## Model Evaluation & Comparison

Models were evaluated using:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- ROC-AUC
- Training time

### Screenshot: Model Comparison Table
<img width="430" height="166" alt="image" src="https://github.com/user-attachments/assets/38a319e2-e445-4ec8-8fb5-05ebf10e83c6" />

**Key Insight:**  
The ensemble model achieved the best overall performance, outperforming all individual models in terms of F1-score and robustness.

---

## Confusion Matrix Analysis

The confusion matrix of the ensemble model shows strong diagonal dominance, indicating correct classification for most ticket categories. Misclassifications mainly occur between semantically similar categories, which is expected in real-world support data.

### 🔹 Screenshot: Confusion Matrix – Ensemble
<img width="925" height="756" alt="image" src="https://github.com/user-attachments/assets/210572b8-d772-4a03-9a24-d6dd3078c30d" />

---

## Ticket Categorization Logic

The system follows a two-stage categorization approach:

1. **ML Prediction:** The ensemble model predicts an operational category.
2. **Semantic Normalization:** Operational labels are mapped to business-friendly categories.

Examples:
- Hardware → Infrastructure Issue
- Administrative rights → Service Request
- Software → Application Issue

---

## Priority Assignment Logic

Priority is assigned using a hybrid approach:

- **Infrastructure Issues** → High
- **Application Issues** → Medium
- **Service Requests** → Low

Confidence-based safety rules:
- Low-confidence predictions are routed for manual review
- Service requests require higher confidence for auto-routing

This ensures critical incidents are escalated quickly while minimizing operational risk.

---

## Routing Decisions

Each ticket is assigned an action:
- **Auto Route** for high-confidence, clear cases
- **Manual Review** for ambiguous or sensitive requests

This design aligns with real enterprise IT support workflows.

---

## End-to-End System Demonstration

The system outputs:
- Raw predicted category
- Final normalized category
- Priority level
- Confidence score
- Routing action

### 🔹 Screenshot: Live Prediction Outputs
<img width="324" height="286" alt="image" src="https://github.com/user-attachments/assets/ea5c1b00-ec3e-48fc-aa8a-bcafd637b9de" />

---

## Business Impact

This system:
- Reduces manual ticket triage workload
- Improves response time for high-priority incidents
- Maintains safety through confidence-aware automation
- Scales efficiently for large support volumes
- Provides interpretable and explainable decisions

---

## Conclusion

This project demonstrates a production-oriented NLP decision-support system rather than a simple text classifier. By combining ensemble learning, semantic understanding, and rule-based logic, the system achieves both strong predictive performance and operational reliability.

**Final Takeaway:**  
The system successfully automates support ticket categorization and prioritization while preserving safety, interpretability, and real-world applicability.

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Sentence-Transformers
- Matplotlib, Seaborn

---

## Future Improvements

- Cross-validation for ensemble weighting
- Integration with ticketing systems (e.g., Jira, ServiceNow)
- Active learning for continuous improvement
- Deployment as a REST API or dashboard

---

## Author

Built as a real-world NLP and machine learning project focused on operational decision support.
