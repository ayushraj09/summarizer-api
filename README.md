# Email Summarization Model

A machine learning model that generates concise summaries of email threads using the BART architecture. The model is deployed as a FastAPI service on Google Cloud Platform.

## Model Details

- **Base Model**: facebook/bart-large-cnn
- **Fine-tuned On**: Email Summary Dataset (xprilion/email-summary-dataset)
- **Training Infrastructure:**
  The model was trained using the following setup:
  - Computing Platform: Kaggle Notebooks
  - Hardware: 2x NVIDIA Tesla T4 GPUs
  - Notebook Link: [View Implementation](https://www.kaggle.com/code/ayushraj911/email-summarizer)
- **Framework**: PyTorch + Transformers

### Performance Metrics

ROUGE scores on the test set:
- ROUGE-1: F1 = 0.469 (R: 0.497, P: 0.450)
- ROUGE-2: F1 = 0.227 (R: 0.244, P: 0.217)
- ROUGE-L: F1 = 0.442 (R: 0.468, P: 0.425)

## Training Process

The model was fine-tuned using the following configuration:
- Batch size: 8 with gradient accumulation steps of 4
- Learning rate: 5e-5 with StepLR scheduler (gamma=0.9)
- Number of epochs: 10 with early stopping
- Maximum input length: 1024 tokens
- Maximum summary length: 128 tokens
- Training optimizations:
  - Gradient clipping
  - Early stopping with patience of 2 epochs
  - Learning rate decay
  - Multi-GPU training support

### Data Preprocessing

The training data underwent several preprocessing steps:
- Cleaning email bodies (removing quoted replies, signatures, disclaimers)
- Formatting metadata (From, To, Subject fields)
- Thread concatenation for multi-email conversations
- Removal of duplicates and invalid entries

## API Service

The model is deployed as a FastAPI service with the following features:

### Endpoints

1. `POST /summarize/`
   - Accepts email text and summarization level
   - Returns generated summary
   - Supports three summarization levels:
     - shortest: 10-20% of input length
     - normal: 20-40% of input length
     - elaborative: 30-60% of input length

2. `GET /`
   - Health check endpoint
   - Returns service status

### Input Validation

- Minimum input length: 10 words
- Maximum input length: 1024 tokens
- Required fields validation
- Error handling for invalid inputs

## Deployment

### GCP Deployment
The model is deployed on Google Cloud Platform with the following components:
- Model artifacts stored in Google Cloud Storage
- FastAPI service for inference
- CORS middleware enabled for API access
- Environment-based configuration

## Requirements

```
fastapi
uvicorn[standard]
torch
transformers
pydantic
google-cloud-storage
enum34
```

## Model Storage

The trained model files are stored in the following structure:
- config.json
- generation_config.json
- merges.txt
- model.safetensors
- special_tokens_map.json
- tokenizer_config.json
- vocab.json

## Future Improvements

Potential areas for enhancement:
1. Add more customization options for summary generation
2. Implement caching for frequently requested summaries

## License

MIT License
