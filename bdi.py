import pandas as pd

bdi = pd.read_csv('C:/Users/Thad/TextMiningProject3/bdi_csv.csv')

def getQuestion(num):
    num = num - 1
    if 0 > num > 21:
        return 'Tell the user the bdi question index is out of range'
    prompt = ''
    row = bdi.iloc[num]
    for i in range(4):
        prompt = prompt + str(i) + ' is ' + row[i+1] + ', ' 
        
    return prompt

def getCategory(num):
    num = num - 1
    if 0 > num > 21:
        return 'Tell the user the bdi question index is out of range'
    row = bdi.iloc[num]
    return row[0]
