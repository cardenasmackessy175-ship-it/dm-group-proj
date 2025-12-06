# Financial News Classification API - MDS5020 Task 2

Flask-based REST API for financial news sentiment analysis and topic classification.

## Project Structure

```
.
├── app.py                          # Flask API application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── .dockerignore                   # Docker ignore patterns
├── Subtask2.1-sentiment_analysis/
│   └── sentiment_model_subtask2.1.joblib
└── subtask2.2_topic_model/
    └── topic_classification_model_subtask2.2.joblib
```

## API Endpoints

### Port
`5724`

### 1. Sentiment Analysis
**Endpoint:** `POST /predict_sentiment`

**Request:**
```json
{
  "news_text": "string"
}
```

**Response:**
```json
{
  "sentiment": "1",      // "1" (positive) or "-1" (negative)
  "probability": "0.98"
}
```

### 2. Topic Classification
**Endpoint:** `POST /predict_topic`

**Request:**
```json
{
  "news_text": "string"
}
```

**Response:**
```json
{
  "topic": "12",         // Topic ID: "1" to "18"
  "probability": "0.63"
}
```

## Topic Labels

| ID | Topic | ID | Topic |
|----|-------|----|-------|
| 1  | IPO Prospectus | 10 | Issuance Prospectus |
| 2  | Sponsor Opinion | 11 | Annual Report (Full) |
| 3  | Articles of Association | 12 | Annual Report (Summary) |
| 4  | Amendment to Articles | 13 | Independent Director Statement |
| 5  | Related Party Transactions | 14 | Director Nomination |
| 6  | Distribution Resolution | 15 | Director Report |
| 7  | Distribution Implementation | 16 | Shareholder Meeting Resolution |
| 8  | Distribution Proposal | 17 | Litigation/Arbitration |
| 9  | Semi-Annual Report | 18 | Executive Personnel Changes |

## Docker Deployment

### Build Image
```bash
docker build -t financial-news-api:latest .
```

### Run Container
```bash
docker run -d \
  -p 5724:5724 \
  --memory="900m" \
  --name financial-api \
  financial-news-api:latest
```

### Check Status
```bash
# View logs
docker logs financial-api

# Check memory usage
docker stats financial-api --no-stream

# Stop container
docker stop financial-api && docker rm financial-api
```

## Technical Constraints

- **Image Size:** ≤ 4GB
- **Runtime Memory:** ≤ 900MB
- **No GPU Access:** Runtime constraint
- **No Internet Access:** Runtime constraint

## Push to Alibaba Cloud Container Registry

```bash
# Login
docker login --username=<username> registry.cn-hangzhou.aliyuncs.com

# Tag image
docker tag financial-news-api:latest \
  registry.cn-hangzhou.aliyuncs.com/<namespace>/financial-news-api:latest

# Push image
docker push registry.cn-hangzhou.aliyuncs.com/<namespace>/financial-news-api:latest
```

**Important:** Ensure repository is set to **PUBLIC** in ACR console.

## Dependencies

- Flask 3.0.0
- scikit-learn 1.4.2
- numpy 1.26.4
- pandas 2.2.2
- joblib 1.4.2
- gunicorn 22.0.0

## Notes

- Models are lazy-loaded on first request to minimize startup time
- Gunicorn uses 2 workers with 2 threads each for optimal performance
- Python 3.9-slim base image for minimal size
