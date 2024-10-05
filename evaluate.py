import trec
import LlamaDetection
import sys
import pandas as pd

#Returns dictionary of {doc_id: rating} for all texts in a .trec file
def trecEval(path):
    results = {}
    texts = trec.parse_trec_file(path)
    for index, row in texts.iterrows():
        results[row['doc_id']] = LlamaDetection.rate(row['text'])
        print(results)
    df = pd.DataFrame(results.items(), columns=['doc_id', 'rating'])
    print(df.head())
    df.to_csv('trec_eval_results.csv')
    return(results)
    
def analyze():
    golden = pd.read_csv('C:/Users/Thad/TextMiningProject3/eRisk2023_T1/g_rels_consenso.csv')
path = 'C:/Users/Thad/TextMiningProject3/eRisk2023_T1/new_data/s_12.trec'
#path=sys.argv[1]
test = trecEval(path)
print(test)
