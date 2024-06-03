from sentence_transformers import SentenceTransformer

model_name = 'sentence-transformers/stsb-roberta-large'
model = SentenceTransformer(model_name)

text = "Water is composed of two hydrogen atoms and one oxygen atom."

embeddings = model.encode(text)

print(embeddings)