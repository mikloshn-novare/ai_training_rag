from ollama import embed, chat

model = "deepseek-r1:14b"

embeddings = embed(model=model, input=["Here is an example sentence I will be embedding!", "Here's a second one!"])

print(len(embeddings['embeddings']))

response = chat(model=model, messages=[
  {
    'role': 'user',
    'content': 'Why did the chicken cross the road?',
  },
])

print(response.message.content)
