# Model and Provider Comparison

| Subtask | Provider | Default model | Reason |
|---|---|---|---|
| Planning | OpenRouter | openai/gpt-4o-mini | Low-cost structured planning; easy model substitution |
| Evidence synthesis | OpenRouter | openai/gpt-4o-mini | Good context handling for evidence cards |
| Review | Groq | llama-3.3-70b-versatile | Separate provider and stronger review-oriented generation |

## Cost
Do not hard-code a dollar figure because provider pricing and model availability can change. During the final demo, record the provider pricing page and the actual model IDs used.

## Latency
Measure application latency with a timer around each API request. Network distance, provider load and token count affect observed latency.

## Context
The RAG pipeline limits the evidence bundle to the highest-scoring retrieved cards to keep prompts bounded and traceable.

## Reasoning
The reviewer is intentionally separated from the writer. This makes it possible to detect unsupported claims before displaying the final result.
