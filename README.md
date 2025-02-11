# EarlyMenAI
Backend for EarlyMenAI playground

## Steps:
1.milvus server
2.get the code from colab into this to make decisions
3.current decisions made in backend:
    >reply message alone
    >reply message + jobs
    >reply message + fetched data
    {tentative}>workflow:::input>fetch data>jobs>message>output


## Instructions on milvus:
https://milvus.io/docs/install_standalone-docker.md

## Curl for chat:
# Basic chat request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Me fire?", "available_actions": ["create_fire", "pick_apple"]}'

# Chat request with custom actions
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What can you do?", "available_actions": ["jump", "run", "sleep"]}'