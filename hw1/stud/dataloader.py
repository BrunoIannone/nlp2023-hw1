import json
def build_dictionary(sentences):
    dict = {}
    idx = 0
    for sentence in sentences:
        for token in sentence:
            if token not in dict:
                dict[token] = idx
                idx+=1

            
    return dict

def build_sentence(file_path):
    f = open(file_path,'r')
    line = f.readline()
    sentences = []
    labels = []
    while (line):
    
        json_line = json.loads(line)
        sentences.append(json_line['tokens'])
        labels.append(json_line['labels'])
        line = f.readline()

    f.close()

    return     {
        'sentences': sentences,
        'labels': labels

    }



