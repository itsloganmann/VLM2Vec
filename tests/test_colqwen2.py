from retriever.colqwen2 import ColQwen2Retriever


def test_colqwen2_dry_run_shapes():
    retriever = ColQwen2Retriever(dry_run=True)
    texts = ["hello", "world"]
    query_out = retriever.embed_query({"texts": texts})
    assert len(query_out.embeddings) == len(texts)
    for tensor in query_out.embeddings:
        assert tensor.shape[-1] == retriever.synthetic_dim

    doc_out = retriever.embed_document({"texts": texts})
    assert len(doc_out.embeddings) == len(texts)
    for tensor in doc_out.embeddings:
        assert tensor.shape[-1] == retriever.synthetic_dim
