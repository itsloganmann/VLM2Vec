from retriever.nemoretriever import NemoRetriever


def test_nemo_dry_run_embeddings():
    retriever = NemoRetriever(dry_run=True)
    batch = {"texts": ["query one", "query two"]}
    outputs = retriever.embed_query(batch)
    assert len(outputs.embeddings) == 2
    for emb in outputs.embeddings:
        assert emb.shape[-1] == retriever.synthetic_dim

    doc_outputs = retriever.embed_document(batch)
    assert len(doc_outputs.embeddings) == 2
    for emb in doc_outputs.embeddings:
        assert emb.shape[-1] == retriever.synthetic_dim
