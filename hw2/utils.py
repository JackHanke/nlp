import json

# 
def read_qa_json(file_name: str, verbose: bool = False):
    answers = ['A','B','C','D']

    data = []
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = ' [CLS] '+ result['fact1'] + ' [SEP] ' + result['question']['stem']

        ans = answers.index(result['answerKey'])
        
        sentences = []
        for j in range(4):
            text = base + ' ' + result['question']['choices'][j]['text'] + ' [SEP] [END]'

            sentences.append(text)

        data.append([sentences, [ans]])

        if verbose:
            # NOTE for debugging
            for sent in sentences:
                print(sent)
            print(' ')
            
            print(result['question']['stem'])
            print(' ',result['question']['choices'][0]['label'],result['question']['choices'][0]['text'])
            print(' ',result['question']['choices'][1]['label'],result['question']['choices'][1]['text'])
            print(' ',result['question']['choices'][2]['label'],result['question']['choices'][2]['text'])
            print(' ',result['question']['choices'][3]['label'],result['question']['choices'][3]['text'])
            print('  Fact: ',result['fact1'])
            print('  Answer: ',result['answerKey'])
            print('  ')
            input()
            
    return data


def read_qa_json_generative(file_name: str, verbose: bool = False):
    answers = ['A','B','C','D']

    data = []
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = ' ' + result['fact1'] + ' ' + result['question']['stem']
        # ans = answers.index(result['answerKey'])

        for j in range(4):
            base += f' [{answers[j]}] ' + result['question']['choices'][j]['text']

        base += ' Answer: ' + result['answerKey']

        data.append(base)

    return data

def read_qa_json_generative_q3(file_name: str, verbose: bool = False):
    answers = ['A','B','C','D']

    data = []
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        
        base = ' ' + result['fact1'] + ' ' + result['question']['stem']
        ans = answers.index(result['answerKey'])

        for j in range(4):
            base += f' [{answers[j]}] ' + result['question']['choices'][j]['text']

        base += ' Answer [' + result['answerKey'] + '] ' + result['question']['choices'][ans]['text'] + ' <|endoftext|>'

        data.append(base)
        
    return data
