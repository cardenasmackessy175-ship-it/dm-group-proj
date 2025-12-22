#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local testing script for the Flask API
Tests both endpoints before deploying to ACR
"""

import requests
import json

# Test configuration
API_BASE_URL = "http://localhost:5724"

# Test data
test_cases = {
    "sentiment": [
        {
            "news_text": "公司业绩持续增长，市场前景看好",
            "expected_sentiment": "1"  # Positive
        },
        {
            "news_text": "公司面临严重亏损，股价大跌",
            "expected_sentiment": "-1"  # Negative
        }
    ],
    "topic": [
        {
            "news_text": "某某公司:2023年度报告摘要",
            "expected_topic": "12"  # Annual Report Summary
        },
        {
            "news_text": "某某公司:独立董事2023年度述职报告",
            "expected_topic": "15"  # Director Report
        }
    ]
}

def test_endpoint(endpoint, data):
    """Test a single endpoint with given data"""
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        response = requests.post(url, json=data, timeout=10)
        print(f"\n{'='*60}")
        print(f"Testing: {endpoint}")
        print(f"Input: {data['news_text'][:50]}...")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return True, result
        else:
            print(f"Error: {response.text}")
            return False, None

    except Exception as e:
        print(f"Exception: {str(e)}")
        return False, None

def main():
    print("="*60)
    print("LOCAL API TESTING")
    print("="*60)
    print("\nMake sure the Flask app is running locally:")
    print("  python app.py")
    print("\nOr run with Docker:")
    print("  docker run -d -p 5724:5724 --name test-api financial-news-api:latest")
    print("="*60)

    # Test sentiment endpoint
    print("\n\n【Testing Sentiment Analysis Endpoint】")
    sentiment_success = 0
    for i, test_case in enumerate(test_cases["sentiment"], 1):
        print(f"\nTest Case {i}:")
        success, result = test_endpoint("predict_sentiment", {"news_text": test_case["news_text"]})
        if success:
            sentiment_success += 1
            if result.get("sentiment") == test_case["expected_sentiment"]:
                print(f"✓ Prediction matches expected: {test_case['expected_sentiment']}")
            else:
                print(f"⚠ Prediction ({result.get('sentiment')}) differs from expected ({test_case['expected_sentiment']})")

    # Test topic endpoint
    print("\n\n【Testing Topic Classification Endpoint】")
    topic_success = 0
    for i, test_case in enumerate(test_cases["topic"], 1):
        print(f"\nTest Case {i}:")
        success, result = test_endpoint("predict_topic", {"news_text": test_case["news_text"]})
        if success:
            topic_success += 1
            if result.get("topic") == test_case["expected_topic"]:
                print(f"✓ Prediction matches expected: {test_case['expected_topic']}")
            else:
                print(f"⚠ Prediction ({result.get('topic')}) differs from expected ({test_case['expected_topic']})")

    # Summary
    print("\n\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Sentiment Tests: {sentiment_success}/{len(test_cases['sentiment'])} passed")
    print(f"Topic Tests: {topic_success}/{len(test_cases['topic'])} passed")

    total_tests = len(test_cases['sentiment']) + len(test_cases['topic'])
    total_success = sentiment_success + topic_success

    if total_success == total_tests:
        print("\n✓ ALL TESTS PASSED! Ready to deploy to ACR.")
    else:
        print(f"\n⚠ {total_tests - total_success} test(s) failed. Please check the errors above.")
    print("="*60)

if __name__ == "__main__":
    main()
