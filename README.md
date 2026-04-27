# AI Inbox Triage Assistant

A Streamlit app that helps a human reviewer triage support-style inbox messages. It classifies the issue, estimates urgency, retrieves matching policy snippets, drafts a reply, checks the result, and logs each run.

## Features

- Message classification into event, schedule, refund, technical, general, complaint, ambiguous, spam, phishing, and malicious-request categories
- Urgency detection and human-review flagging
- TF-IDF retrieval over a local JSON knowledge base
- Grouped example library for real requests, ambiguous requests, spam, and malicious/security cases
- OpenAI-backed classification, response drafting, and checking when an API key is provided
- Local deterministic fallback so the full workflow still runs without an API key
- CSV logging for transparency and debugging
- Labeled evaluation script for course demo metrics

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You can provide your OpenAI API key in either place:

- Paste it into the Streamlit sidebar when the app starts.
- Or create a local `.env` file from `.env.example` and set `OPENAI_API_KEY`.

The app does not write the API key to logs.

## Run The App

```bash
streamlit run app.py
```

Then paste an incoming message or choose one of the grouped sample messages. Results are saved to `logs/app_logs.csv`.

The example library is organized into:

- `Real Requests`
- `Ambiguous Requests`
- `Spam Requests`
- `Malicious/Security Requests`

## Run Evaluation

Local fallback evaluation:

```bash
python eval/evaluate.py
```

OpenAI evaluation after setting `OPENAI_API_KEY`:

```bash
python eval/evaluate.py --use-openai
```

Optional detailed output:

```bash
python eval/evaluate.py --output logs/evaluation_results.csv
```

## Project Structure

```text
app.py                   Streamlit reviewer UI
data/faq.json            Local knowledge base
data/sample_messages.csv Sample messages for the UI
data/test_set.csv        Labeled examples for evaluation
eval/evaluate.py         Evaluation runner
src/classifier.py        Category and urgency classifier
src/retriever.py         TF-IDF knowledge retriever
src/generator.py         Draft response generator
src/checker.py           Guardrail and review checker
src/orchestrator.py      End-to-end workflow controller
src/logger.py            CSV logger
```

## Notes For Testing

The default model is `gpt-5.4-mini`, but you can change the model name in the Streamlit sidebar or with `OPENAI_MODEL` in `.env`. If no API key is provided, the app automatically uses local keyword rules and templates so you can still demonstrate the workflow.
