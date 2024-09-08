import ollama
olama_models = ["llama3.1", "gemma2", "phi3"]


def chat_with_ollama(msg, model_name):
    stream = ollama.chat(
        model=model_name,
        messages=[
            {'role': 'system', 'content': 'You are a question-answer helper. Analyze the question and provide a concise answer.'},
            {'role': 'system', 'content': 'Response have to be in no more than 15 words.'},             
            {'role': 'user', 'content': msg}
        ],
        stream=True
    )    
    res = []
    for chunk in stream:        
        res.append(chunk['message']['content'])
        #print(chunk['message']['content'], end='', flush=True)
    return (chunk['model'], ''.join(res))    

def exec_msg_with_cascade_of_models(msg):
    response = []
    for mod_name in olama_models:
        response.append(chat_with_ollama(msg, mod_name))
    return response    
        
if __name__ == "__main__":
    res = exec_msg_with_cascade_of_models("Who is Socrates?")
    
    for r in res:        
        print(r)    