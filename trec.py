from trectools import TrecQrel, procedures

def openTrec(path):
    qrels = TrecQrel(path)
    runs = procedures.list_of_runs_from_path(qrels, "*.gz")
    print(runs)
openTrec('C:/Users/Thad/Downloads/t1_training_collection_2024/t1_training/TRAINING DATA (2023 COLLECTION)/eRisk2023_T1/g_qrels_majority_2.csv')
