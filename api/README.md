Steps to start the API server

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
