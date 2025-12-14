Steps to start the API server
```shell
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
uvicorn app:app --reload
```
Steps to test the API server"
```shell    
curl -s -X POST "http://127.0.0.1:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"how are you today"}' | jq
```

To commit and push changes to the repository, use the following commands:
```shell
git add .
git commit -m "changes"
git push
```