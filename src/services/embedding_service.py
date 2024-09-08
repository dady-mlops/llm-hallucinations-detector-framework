import torch
from transformers import BertTokenizer, BertModel

def text_to_embedding(text):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Load pre-trained model
    model = BertModel.from_pretrained('bert-base-uncased')

    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        # Perform forward pass (feed input to the model)
        outputs = model(**inputs)

    # The last hidden state is typically used as the embeddings
    last_hidden_states = outputs.last_hidden_state

    # Pooling method: mean of the last hidden states
    embeddings = torch.mean(last_hidden_states, dim=1)

    return embeddings

def cosine_similarities(real_answers: str, llm_answers):
    return None

def vector_similarity_llm_answer_real_answer(processed_data):    
    for _, entry in enumerate(processed_data):                
        for qa in entry['qas']:                        
            real_answers = qa['answers'][0]['text']
            #TODO 
            llm_answers = qa['llms_answer'] 
    return processed_data 
    