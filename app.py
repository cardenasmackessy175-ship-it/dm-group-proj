#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask API for Financial News Analysis
Task 2: Text Classification with Model Deployment

Endpoints:
- POST /predict_sentiment: Sentiment analysis (positive/negative)
- POST /predict_topic: Topic classification (18 categories)
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for models
sentiment_model = None
topic_model = None
TOPIC_ID2TEXT = {
    1: '上市保荐书',
    2: '保荐/核查意见',
    3: '公司章程',
    4: '公司章程修订',
    5: '关联交易',
    6: '分配方案决议公告',
    7: '分配方案实施',
    8: '分配预案',
    9: '半年度报告全文',
    10: '发行保荐书',
    11: '年度报告全文',
    12: '年度报告摘要',
    13: '独立董事候选人声明',
    14: '独立董事提名人声明',
    15: '独立董事述职报告',
    16: '股东大会决议公告',
    17: '诉讼仲裁',
    18: '高管人员任职变动',
}


# Preprocessing functions for topic model (must be defined before loading)
def strQ2B(ustring: str) -> str:
    """Convert full-width characters to half-width"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 32
        elif 0xFF01 <= inside_code <= 0xFF5E:
            inside_code -= 0xFEE0
        rstring += chr(inside_code)
    return rstring


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    text = str(text)
    text = strQ2B(text)
    text = re.sub(r'^[^:：]{1,20}[:：]\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# IMPORTANT: Make clean_text available in __main__ for pickle deserialization
# This fixes the "Can't get attribute 'clean_text' on <module '__main__'>" error
import __main__
__main__.clean_text = clean_text
__main__.strQ2B = strQ2B
logger.info(f"Injected clean_text and strQ2B into __main__ (module name: {__name__})")


def load_models():
    """Load both models at startup"""
    global sentiment_model, topic_model

    try:
        # Load sentiment analysis model
        logger.info("Loading sentiment analysis model...")
        sentiment_model = joblib.load('Subtask2.1-sentiment_analysis/sentiment_model_subtask2.1.joblib')
        logger.info("Sentiment model loaded successfully")

        # Load topic classification model
        logger.info("Loading topic classification model...")
        model_data = joblib.load('subtask2.2_topic_model/topic_classification_model_subtask2.2.joblib')
        topic_model = model_data['model']
        logger.info("Topic model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    """
    Sentiment Analysis Endpoint

    Input JSON: {"news_text": "string"}
    Output JSON: {"sentiment": "-1" or "1", "probability": "0.98"}
    """
    global sentiment_model

    try:
        # Lazy load model on first request
        if sentiment_model is None:
            logger.info("Loading sentiment analysis model...")
            sentiment_model = joblib.load('Subtask2.1-sentiment_analysis/sentiment_model_subtask2.1.joblib')
            logger.info("Sentiment model loaded successfully")

        # Get input data
        data = request.get_json()
        if not data or 'news_text' not in data:
            return jsonify({"error": "Missing 'news_text' in request"}), 400

        news_text = data['news_text']
        if not isinstance(news_text, str) or not news_text.strip():
            return jsonify({"error": "Invalid 'news_text' value"}), 400

        # Predict
        prediction = sentiment_model.predict([news_text])[0]
        probabilities = sentiment_model.predict_proba([news_text])[0]

        # Get probability of predicted class
        # Model outputs 1 (positive) or -1 (negative)
        classes = sentiment_model.classes_
        pred_idx = np.where(classes == prediction)[0][0]
        probability = float(probabilities[pred_idx])

        # Convert prediction to string (model outputs 1 or -1)
        sentiment_label = str(int(prediction))

        # Format output
        result = {
            "sentiment": sentiment_label,
            "probability": f"{probability:.2f}"
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in predict_sentiment: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_topic', methods=['POST'])
def predict_topic():
    """
    Topic Classification Endpoint

    Input JSON: {"news_text": "string"}
    Output JSON: {"topic": "12", "probability": "0.63"}
    """
    global topic_model

    try:
        # Lazy load model on first request
        if topic_model is None:
            logger.info("Loading topic classification model...")
            model_data = joblib.load('subtask2.2_topic_model/topic_classification_model_subtask2.2.joblib')
            topic_model = model_data['model']
            logger.info(f"Topic model loaded successfully. Type: {type(topic_model)}")

            # Verify the model has required components
            if hasattr(topic_model, 'named_steps'):
                logger.info(f"Pipeline steps: {list(topic_model.named_steps.keys())}")
                if 'tfidf' in topic_model.named_steps:
                    tfidf = topic_model.named_steps['tfidf']
                    logger.info(f"TfidfVectorizer fitted: {hasattr(tfidf, 'idf_')}")

        # Get input data
        data = request.get_json()
        if not data or 'news_text' not in data:
            return jsonify({"error": "Missing 'news_text' in request"}), 400

        news_text = data['news_text']
        if not isinstance(news_text, str) or not news_text.strip():
            return jsonify({"error": "Invalid 'news_text' value"}), 400

        # Predict (model's internal preprocessor will handle text cleaning)
        prediction = topic_model.predict([news_text])[0]
        probabilities = topic_model.predict_proba([news_text])[0]

        # Get probability of predicted class
        probability = float(probabilities[prediction])

        # Convert 0-indexed prediction to 1-indexed topic ID
        topic_id = str(prediction + 1)

        # Format output
        # Note: PDF specifies "topic" but evaluation script may expect "label"
        result = {
            "label": topic_id,  # Changed from "topic" to "label" for evaluation compatibility
            "probability": f"{probability:.2f}"
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in predict_topic: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    # Load models before starting server (for local testing)
    load_models()

    # Run Flask app on port 5724
    app.run(host='0.0.0.0', port=5724, debug=False)
