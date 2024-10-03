from huggingface_hub import login
login('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')

import bdi
import transformers
import torch
import pandas as pd
import re 

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

#Helper method to extract integer value from Llama output
def extract_integer(data):
    # Check if 'content' key exists and is a string
    if 'content' in data and isinstance(data['content'], str):
        # Search for an integer in the 'content' value
        match = re.search(r'\d+', data['content'])
        if match:
            return int(match.group())
    return -1

def predict(statement):
    statement = str(statement)
    #Initialize output matrix as dictionary
    output = {'Catagory': [],
            'Rating': []}
    #Initialize output data frame
    eval = pd.DataFrame(output)
    print('Evaluating statement..')
    #Evaluate each catagory
    for i in range(1, 22):
        messages = [
        {"role": "system", "content": "You're going to read a social media post and return only the integer index corrosponding to the most relevant of these statements. Do not return anything other than the appropriate integer index. An example of a valid answer is \"0\", \"1\", \"2\", or \"3\", an example of an invalid answer would be \"Based on the user statement, the most relevant index is 3\". If no statments are applicable, default to \"0\". Here are the statements: " + bdi.getQuestion(i)},
        {"role": "user", "content": statement},
        ]
        outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        )
        print('Evaluating question '+ str(i) + "/21.")
        raw =  (outputs[0]["generated_text"][-1])
        result = raw['content']
        new_row = {'Catagory': [str(bdi.getCategory(i))], 'Rating': [str(result)]}
        new_row = pd.DataFrame(new_row)
        eval = eval._append(new_row, ignore_index = False)
        print(new_row)
    return eval

result = predict("Today my day was very busy. My friend thinks that one of the conference men was hitting on me. I did not believe her until she mentioned that he called me a model oops. Busy day, I'm scare of people and being an intern is boring. Rip")
result.to_csv('BDI_single_eval4.csv', index=False) 
print(result.head())




