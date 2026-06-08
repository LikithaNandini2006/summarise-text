# 📝 AI Text Summarization System

An AI-powered Text Summarization Web Application developed using Python, Flask, Scikit-Learn, Fuzzy Logic, HTML, and CSS. This application helps users summarize lengthy text into a shorter and meaningful version by extracting the most important sentences.

---

## 📌 Project Description

Reading large documents, articles, reports, or paragraphs can be time-consuming. This project provides an intelligent solution by automatically generating concise summaries while preserving the core meaning of the original text.

The application uses Natural Language Processing (NLP) techniques such as TF-IDF Vectorization, Cosine Similarity, and Fuzzy Logic-based sentence scoring to identify and extract the most relevant sentences from the input text.

Users simply paste a large block of text into the application, click the **Summarize** button, and instantly receive a summarized version.

---

## 🚀 Features

- Summarizes long paragraphs and documents
- Extractive text summarization
- TF-IDF based sentence ranking
- Cosine similarity calculation
- Fuzzy Logic sentence scoring
- User-friendly web interface
- Fast and efficient processing
- Responsive design
- Clean and attractive UI
- Easy to use

---

## 🛠️ Technologies Used

### Backend
- Python
- Flask

### Machine Learning & NLP
- Scikit-Learn
- NumPy
- RapidFuzz (Fuzzy Logic)

### Frontend
- HTML5
- CSS3

---

## 📂 Project Structure

```text
summarisation/
│
├── app.py
├── summarizer.py
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── screenshots/
    ├── home.png
    └── output.png
```

---

## ⚙️ Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/text-summarization-system.git
```

### Step 2: Navigate to Project Directory

```bash
cd text-summarization-system
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 🧠 How It Works

### 1. Text Input
The user enters or pastes a large paragraph into the input box.

### 2. Sentence Segmentation
The text is split into individual sentences.

### 3. TF-IDF Vectorization
Each sentence is converted into numerical vectors using TF-IDF.

### 4. Cosine Similarity
The similarity between sentences is calculated.

### 5. Fuzzy Logic Scoring
Fuzzy matching is used to determine sentence relevance.

### 6. Sentence Ranking
Sentences are ranked based on combined scores.

### 7. Summary Generation
The highest-ranked sentences are selected.

### 8. Output Display
The summarized text is displayed in the output section.

---

## 📸 Screenshots


### Summary Output

![Summary Output](<img width="1366" height="768" alt="Screenshot (14)" src="https://github.com/user-attachments/assets/ccdbedc6-0c2e-4bdc-8cfb-d7bf2cd09176" />
)

---

## 📊 Sample Input

Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including object-oriented, procedural, and functional programming. Python is widely used in web development, machine learning, artificial intelligence, data science, automation, and software development. Its large ecosystem of libraries and frameworks makes development faster and more efficient.

---

## 📋 Sample Output

Python is a high-level programming language known for its simplicity and readability. It is widely used in web development, machine learning, artificial intelligence, data science, automation, and software development.

---

## 🎯 Applications

- Document Summarization
- News Article Summarization
- Research Paper Summarization
- Educational Content Summarization
- Blog Content Summarization
- Report Summarization
- Content Analysis

---

## 🔮 Future Enhancements

- Transformer-based Summarization
- PDF File Summarization
- DOCX File Summarization
- Multi-language Support
- Speech-to-Text Summarization
- Text-to-Speech Conversion
- User Authentication
- Download Summary as PDF
- Cloud Deployment

---

## 📚 Learning Outcomes

This project demonstrates practical implementation of:

- Natural Language Processing (NLP)
- Machine Learning Concepts
- Information Retrieval
- Text Mining
- Fuzzy Logic
- Flask Web Development
- Frontend Design
- Data Processing Techniques

---

## 👨‍💻 Author

**Likitha Nandini**

GitHub: https://github.com/LikithaNandini2006

---

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome. Feel free to fork this repository and submit a pull request.

---

## 📄 License

This project is developed for educational and learning purposes.
