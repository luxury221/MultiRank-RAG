# PDF Input Directory

Place local PDF files here when running the offline pipeline.

```bash
python scripts/06_run_pipeline.py --questions data/questions.csv
```

PDF files in this directory are ignored by Git because they may contain private data or large benchmark files.

If automatic parsing misses important tables, captions or figures, add curated evidence to `data/manual_nodes.csv` and rerun the pipeline.

