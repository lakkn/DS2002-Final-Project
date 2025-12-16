# DS2002-Final-Project
Lakshay Kansal (cjm2gm), Micaiah Lee (jhq5mr), Peter Cao (etm9uh)

How to Build FAISS Index/ Rag Pipeline

1. Create a Python environment and install the packages in `rag_pipline/requirements.txt` and `api/requirements.txt`
2. Build the FAISS Index with `python -m rag_pipeline.rag --build-index --data-dir data/csv_data`
3. Steps to start the API server:

```shell
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
uvicorn api.app:app --reload
```

Steps to test the API server"
```shell
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What trends do we see in COVID cases in America?"}'
```

To commit and push changes to the repository, use the following commands:
```shell
git add .
git commit -m "changes"
git push
```

