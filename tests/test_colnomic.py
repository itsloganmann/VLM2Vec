from retriever.colnomic import ColNomicRetriever


def test_colnomic_multi_vector_dry_run():
    retriever = ColNomicRetriever(dry_run=True, output_format="multi")
    batch = {"texts": ["query one", "query two"]}
    outputs = retriever.embed_query(batch)
    assert len(outputs.embeddings) == 2
    for emb in outputs.embeddings:
        assert emb.ndim == 2


def test_colnomic_single_vector_dry_run():
    retriever = ColNomicRetriever(dry_run=True, output_format="single")
    batch = {"texts": ["only"]}
    outputs = retriever.embed_document(batch)
    assert len(outputs.embeddings) == 1
    assert outputs.embeddings[0].ndim == 1
