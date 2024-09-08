'''
The Natural Questions (NQ) dataset is a large-scale question-answering benchmark 
that is primarily used for training and evaluating machine learning models. 
It consists of real-world questions from Google search and their corresponding answers based on Wikipedia pages.
'''

import csv
import json
from typing import List, Dict


class QuestionEntity:
    def __init__(self, question, long_answers, short_answers):
        self.question = question
        self.long_answers = long_answers
        self.short_answers = short_answers

    def __repr__(self):
        return f"QuestionEntity(question='{self.question}', long_answers='{self.long_answers}', short_answers='{self.short_answers}')"


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
                    question=row['question'],
                    long_answers=row['long_answers'],
                    short_answers=row['short_answers']
                )
                entities.append(entity)
        return entities