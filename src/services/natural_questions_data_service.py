'''
The Natural Questions (NQ) dataset is a large-scale question-answering benchmark 
that is primarily used for training and evaluating machine learning models. 
It consists of real-world questions from Google search and their corresponding answers based on Wikipedia pages.
'''
import os
import pandas as pd
import csv
from typing import List, Dict
import uuid

class QuestionEntity:
    def __init__(self, uid, question, long_answer):
        self.uid = uid
        self.question = question
        self.long_answer = long_answer
        self.llm_answers = {}
        self.long_answer_similarity = {}

    def __repr__(self):        
        return f"QuestionEntity(uid='{self.uid}', question='{self.question}', long_answer='{self.long_answer}')"
    
    def __str__(self):        
        return (
            f"UID: {self.uid}\n"
            f"Question: {self.question}\n"
            f"Long Answer: {self.long_answer}\n"
            f"LLM Answers: {self.llm_answers}\n"
            f"Long Answer Similarity: {self.long_answer_similarity}\n"
        )


class NaturalQuestionsParser:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse_entities(self, n=None):
        """
        Parse the CSV file and return a list of QuestionEntity objects.

        :param n: Number of rows to read from the CSV. If None, read all rows.
        :return: List of QuestionEntity objects.
        """
        entities = []
        with open(self.file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                if n is not None and i >= n:
                    break
                entity = QuestionEntity(
                    uid=row['uid'],
                    question=row['question'],
                    long_answer=row['long_answer'],                    
                )
                entities.append(entity)
        return entities
    
    def delete_empty_or_very_long_short_answs(self):
        """
        Updates the dataset by removing rows where `long_answer` is empty, has fewer than 10 words, or more than 80 words.
        """
        rows_to_keep = []
        
        # Read the file and filter rows based on the conditions
        with open(self.file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames  # Keep field names for rewriting
            for row in reader:
                long_answer = row['long_answer']
                word_count = len(long_answer.split())
                
                # Check if long_answer is non-empty and has between 10 and 80 words
                if long_answer and 10 <= word_count <= 80:
                    rows_to_keep.append(row)

        # Overwrite the original file with filtered rows
        with open(self.file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_keep)
            
    def add_uid_if_empty(self):
        df = pd.read_csv(self.file_path)

        # Function to generate UID
        def generate_uid():
            return str(uuid.uuid4())

        # Apply UID generation where 'UID' column is NaN or empty
        df['uid'] = df['uid'].apply(lambda x: generate_uid() if pd.isna(x) or x == '' else x)

        # Save the updated dataframe back to CSV
        df.to_csv(self.file_path, index=False)

        return df


    def store_llms_responses(self, entities: list, res_file_path: str):
        # Prepare a list of dictionaries for storing rows
        data = []
        all_llm_keys = set()

        # Collect all unique LLM answer keys (column names)
        for entity in entities:
            all_llm_keys.update(entity.llm_answers.keys())

        # For each entity, create a dictionary row
        for entity in entities:
            row = {
                "uid": entity.uid,
                "question": entity.question,
                "long_answer": entity.long_answer,
            }
            # Fill in the llm_answers for each key, if present
            for key in all_llm_keys:
                row[key] = entity.llm_answers.get(key, "")
            data.append(row)
        
        # Convert the list of dictionaries to a DataFrame
        new_df = pd.DataFrame(data)
        
        # Check if the CSV file already exists
        if os.path.exists(res_file_path):
            # Read the existing CSV file
            existing_df = pd.read_csv(res_file_path)
            
            # Filter out entities that already exist (based on 'uid')
            new_df = new_df[~new_df['uid'].isin(existing_df['uid'])]

            # Append the new records to the existing dataframe
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            # If no file exists, just use the new dataframe
            combined_df = new_df

        # Save the updated dataframe to the CSV file
        combined_df.to_csv(res_file_path, index=False)

        return combined_df
    
    
    def read_llm_results_from_csv(self):
        entities = []

        with open(self.file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                # Extract uid, question, and long_answer
                uid = row['uid']
                question = row['question']
                long_answer = row['long_answer']
                
                # Create QuestionEntity instance
                entity = QuestionEntity(uid, question, long_answer)
                
                # Extract additional LLM answers (phi3, gemma2, llama3.1, etc.)
                for key, value in row.items():
                    if key not in ['uid', 'question', 'long_answer']:
                        entity.llm_answers[key] = value
                
                # Add entity to the list
                entities.append(entity)
        
        return entities