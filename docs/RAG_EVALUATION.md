# RAG Evaluation Protocol

Five fixed queries are provided in `tests/retrieval_eval.json`.

For each query:
1. Retrieve top 5 documents.
2. Compare the returned IDs against the expected relevant IDs.
3. Calculate Precision@5.
4. Manually inspect relevance and note false positives/false negatives.
5. Repeat after corpus updates to prevent accidental regression.

## Important interpretation

The benchmark is a small coursework evaluation, not a statistically comprehensive retrieval study. A high score does not prove the corpus is complete, and a low score can indicate query vocabulary mismatch rather than poor semantic relevance.
