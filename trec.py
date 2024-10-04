import pandas as pd
from ranx import Qrels, Run

def openTrec(path):
    raw = pd.read_csv(path)
    raw['query'] = raw['query'].map(str)
    qrels = Qrels.from_df(
    df=raw,
    q_id_col="query",
    doc_id_col="docid",
    score_col="rel",
    )
    print(qrels)

def parse_trec_file(trec_file_path):
    """
    Parses a TREC file and returns a dictionary that maps doc_id to document text.
    """
    with open(trec_file_path, 'r') as f:
        trec_data = f.read()

    doc_texts = {}
    docs = trec_data.split("</DOC>")  # Split by DOC closing tag

    for doc in docs:
        if "<DOC>" in doc:
            # Extract doc_id and document text
            doc_id_start = doc.find("<DOCNO>") + len("<DOCNO>")
            doc_id_end = doc.find("</DOCNO>")
            doc_id = doc[doc_id_start:doc_id_end].strip()

            # Get the document content by removing tags
            doc_content_start = doc.find("<TEXT>") + len("<TEXT>")
            doc_id_end = doc.find("</TEXT>")
            doc_text = doc[doc_content_start:doc_id_end].strip() 

            doc_texts[doc_id] = doc_text
    texts = pd.DataFrame(doc_texts.items(), columns=['doc_id', 'text'])
    return texts

