from sentence_transformers import util


def find_best_chunks(question, chunks, chunk_embeddings, model, top_k=3):
    question_embedding = model.encode(question)

    results = util.semantic_search(
        question_embedding,
        chunk_embeddings,
        top_k=top_k
    )

    best_chunks = []

    for result in results[0]:
        index = result["corpus_id"]
        score = result["score"]

        best_chunks.append({
            "chunk": chunks[index],
            "score": score
        })

    return best_chunks