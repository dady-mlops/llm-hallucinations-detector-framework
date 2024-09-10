from typing import List
import ollama
import ollama

# RUN the next models wiht OLLAMA - https://ollama.com/library
LLM_MODELS = {"llama3.1", "gemma2", "phi3"}

class OllamaChat:
    def __init__(self, models=None):
        if models is None:
            self.models = LLM_MODELS
        else:
            self.models = models

    def chat_with_ollama(self, model_name, msg):
        stream = ollama.chat(
            model=model_name,            
            messages=[
                {'role': 'system', 'content': 'You are a question-answer helper. Analyze the question and provide a concise answer.'},
                {'role': 'system', 'content': 'Response have to be in no more than 60 words.'},
                {'role': 'user', 'content': msg}
            ],
            stream=True
        )
        res = []
        for chunk in stream:
            res.append(chunk['message']['content'])
        return ''.join(res)

    def exec_msg_with_cascade_of_models(self, msgEntities: List):                      
        for msgEntity in msgEntities:
            res_dir = {}
            for mod_name in self.models:                                  
                res_dir[mod_name] = self.chat_with_ollama(mod_name, msgEntity.question)
            msgEntity.llm_answers = res_dir     
                
        return msgEntities 