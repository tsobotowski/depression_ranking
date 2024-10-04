import trec
import LlamaDetection
import sys
import pandas as pd

#Returns dictionary of {doc_id: rating} for all texts in a .trec file
def trecEval(path):
    results = {}
    texts = trec.parse_trec_file(path)
    results = texts['doc_id']
    for index, row in texts.iterrows():
        results[row['doc_id']] = LlamaDetection.rate(row['text'])
    df = pd.DataFrame(results)
    df.to_csv('trec_eval_results.csv')
    return(results)
    
path = 'C:/Users/Thad/TextMiningProject3/eRisk2023_T1/new_data/test.trec'
path=sys.argv[1]
test = trecEval(path)
print(test)
