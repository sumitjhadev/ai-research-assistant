# Alternative arXiv Query Topics

The default ingestion topic for this project is **"retrieval augmented generation"**.
The pipeline is fully generic, so you can re-point it at any arXiv-searchable topic
by re-running ingestion with a different `--query`. A few ideas to repurpose this
project for a different research area:

```bash
python backend/ingest.py --query "LLM agents" --max-results 50
python backend/ingest.py --query "vector database benchmarking" --max-results 50
python backend/ingest.py --query "transformer efficiency" --max-results 50
python backend/ingest.py --query "multimodal foundation models" --max-results 50
python backend/ingest.py --query "reinforcement learning from human feedback" --max-results 50
```

After changing topics, remember to:

1. Re-run `python backend/vectorstore.py build` to rebuild the FAISS index over the new corpus.
2. Update `data/research_questions.json` with questions relevant to the new topic
   (or regenerate a fresh eval set) and fill in `expected_sources` after manually
   verifying which papers actually answer each question.
3. Restart the FastAPI backend and Streamlit frontend so they pick up the new corpus.

## Topic ideas and why they work well

| Query | Why it's a good fit |
|---|---|
| `LLM agents` | Fast-moving subfield with lots of recent papers on tool use, planning, and multi-agent systems — good for testing comparison across very different architectures. |
| `vector database benchmarking` | Narrower, more technical corpus — good for testing precision of retrieval on numeric/benchmark-heavy content. |
| `transformer efficiency` | Covers sparse attention, quantization, distillation — good for testing the comparison table's "Method" and "Results" columns across very different optimization techniques. |
| `multimodal foundation models` | Broader corpus spanning vision-language models — good stress test for the citation-grounding guardrail on more heterogeneous content. |
| `reinforcement learning from human feedback` | Well-studied alignment subfield with clear methodology/dataset/results structure — good for the paper summarization feature. |
