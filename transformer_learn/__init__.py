
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens',
                                      device='mps')

a=["asdasdasd asdiwjvja"]
print(embedding_model.encode(a))
