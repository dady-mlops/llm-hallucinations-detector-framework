def gemma():    
    return "gema", "gemma mock answ"

def lama():    
    return "lama", "lama mock answ"

def phi():    
    return "phi", "phi mock answ"


def get_llm_response():
    res = []
    res.append(gemma())
    res.append(lama())
    res.append(phi())
    return res


def questions_to_answers_with_llms(processed_data):
    for _, entry in enumerate(processed_data):        
        for qa in entry['qas']:
            print(f"  Q: {qa['question']}")
            print(f"  A: {qa['answers'][0]['text']}")
            qa['llms_answer'] = get_llm_response()
    return processed_data        
            
            