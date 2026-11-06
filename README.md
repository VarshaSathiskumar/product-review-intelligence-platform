# 🛍️ Product Review Intelligence Platform

## 📌 Overview
An AI-powered web application that analyzes unstructured product review data to generate meaningful insights such as sentiment trends, review summaries, pros and cons, customer complaints, and product comparisons.

Users can:
- Analyze Kaggle product review datasets
- Explore interactive dashboards and AI-generated insights

---

# 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend / Web App | Streamlit | Builds the interactive web application and dashboard UI |
| Backend APIs | FastAPI | Handles backend API logic and model integration |
| Programming Language | Python | Core language for NLP, data processing, and backend development |
| Database | PostgreSQL | Stores application and processed review data |
| Models | Llama 3.1, BGE-M3, and BERTopic | Handles LLM-based analysis, multilingual embeddings, and topic modeling
| Visualisation | Apache Superset | Builds BI-style dashboards and trend analytics |
| Deployment | Streamlit Community Cloud | Hosts the interactive web application |

---

# 🤖 AI Models

| Model | Runtime | Used for |
|---|---|---|
| Llama 3.1 8B (`llama3.1:8b`) | Ollama (local) | Sentiment analysis (positive / neutral / negative with confidence, complaint flagging) and pros/cons review summarisation |
| BGE-M3 (`BAAI/bge-m3`) | sentence-transformers | Semantic embeddings of reviews and summaries (used for topic discovery and summary relevance checks) |
| BERTopic | Python library (UMAP + HDBSCAN) on BGE-M3 embeddings | Discovering recurring customer complaints per product |

Setup: install Ollama, run `ollama pull llama3.1:8b`, then `pip install -r requirements.txt`. BGE-M3 is downloaded automatically on first use.

---

# 🚀 Core Features

- CSV review dataset upload
- Kaggle dataset support
- Sentiment analysis
- AI-generated review summaries
- Pros and cons extraction
- Customer complaint detection
- Semantic review search
- Product comparison analytics
- Interactive dashboards and visualizations
- AI review analyst agent

---

# 🌐 Deployment

The application will be:

- Developed locally using Python and Jupyter notebooks
- Version controlled using GitHub
- Deployed using Streamlit Community Cloud
- Containerized using Docker for scalability

---

# 📊 Results

## Method and sampling

Scoring every review with Llama 3.1 locally is slow, so the evaluation uses a sample of about 3,000 reviews across 6 categories (Apparel, Beauty, Books, Furniture, Mobile, Outdoors). Ground truth for sentiment is derived from star ratings (1–2 → negative, 3 → neutral, 4–5 → positive), so it is a proxy for sentiment, not human labels.

- **Analysis is per product.** Pros/cons summaries and recurring-complaint topics are produced for each individual product. To make that meaningful, the sample takes the **5 most-reviewed products in each category and up to 100 reviews of each**, rather than a random sample.
- **Why not a random sample:** a first run used a random 500 reviews per category. Reviews were spread over about 2,300 products, so almost no product had more than one or two complaints. Per-product complaint topics were not possible (only 4–5 products qualified), so that run grouped complaints by **product category** instead. That category-level view (`complaint_topics_by_category.parquet`) is still produced as an extra, but the main output is per product.
- Recurring-complaint topics are discovered once over all sampled complaints (BERTopic needs a reasonably large corpus) and then summarised per product.

**Evaluation sample:** 30 products (the 5 most-reviewed per category), 1,652 reviews.

## Sentiment analysis (Llama 3.1 8B)

| Metric | Score |
|---|---|
| Accuracy | 0.881 |
| Weighted precision | 0.911 |
| Weighted recall | 0.881 |
| Weighted F1 | 0.890 |
| Macro F1 | 0.69 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Positive | 0.99 | 0.89 | 0.94 | 1,318 |
| Neutral | 0.28 | 0.38 | 0.32 | 81 |
| Negative | 0.70 | 0.98 | 0.82 | 253 |

- Weighted F1 by category: Outdoors 0.954, Mobile 0.923, Apparel 0.908, Furniture 0.894, Beauty 0.889, Books 0.750.
- The model is strong on clearly positive and negative reviews and weak on neutral (3-star) ones, which it often labels negative or positive. 3-star reviews often mix praise and complaints, so the star proxy is noisy there.
- 291 of the 1,652 reviews (17.6%) are flagged as high-confidence complaints (negative label with confidence ≥ 0.85).

## Pros/cons summaries (Llama 3.1)

27 products (those with at least one positive and one negative review) were summarised. Summaries have no reference text, so quality is checked against the source reviews:

| Metric | Pros | Cons |
|---|---|---|
| Relevance (BGE-M3 cosine similarity to source reviews) | 0.791 | 0.749 |
| Coverage (ROUGE-1 recall) | 0.142 | 0.196 |
| Conciseness (summary length ÷ source length, mean / median) | 0.26 / 0.19 | 1.01 / 0.29 |

- Relevance is good: the summaries stay on topic. ROUGE recall is low, as expected for abstractive summaries that paraphrase rather than copy.
- The mean conciseness for cons is above 1 because some products have only one short negative review, so the summary is longer than its source. The median (0.29) is the more representative figure.

## Recurring complaints (BGE-M3 + BERTopic)

- 15 complaint topics were found across the sample, such as app permissions and privacy, mattress and bed frame problems, hat and clothing sizing, and specific book criticisms.
- 16 of the 30 products had at least 5 complaints and get per-product topics (35 product-topic rows). A category-level view is also produced.
- Quality checks on 12 products with at least two topics: average silhouette score 0.134 (only 2 of 12 above 0.2, so topics overlap somewhat in embedding space), top topics cover 99% of each product's complaints, and keyword distinctiveness between a product's topics is 0.95.

---

# 🎯 Expected Outcomes

- Summarize customer reviews
- Detect positive and negative feedback
- Identify recurring customer complaints
- Provide sentiment and trend analysis
- Help customers make informed purchasing decisions
- Help sellers improve products based on customer feedback
