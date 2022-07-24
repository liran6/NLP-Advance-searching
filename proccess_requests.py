# Imports
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class JobSimilarityClass:

    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens') #using GOOGLE-bert model

    def get_best_jobs(self, job_titles, job_ids, user_prompt, k=5):

        encoded_jobs = self.model.encode(job_titles)# transform each title to a vector size of 768 -to ask madar
        encoded_kw = self.model.encode([user_prompt])# transform the user search text to a vector size of 768

        # [(similarity, title, id) .... ]
        result_list = []
        for idx, encoded_job in enumerate(encoded_jobs):
            cosine_sim = cosine_similarity(encoded_kw, [encoded_job])[0] # cross the user search with a titlle- return confidence
            result_list.append((cosine_sim, job_titles[idx], job_ids[idx]))

        sorted_res = sorted(result_list, key=lambda tup: tup[0], reverse=True)#sort all the jobs by the model score
        best_fits = sorted_res[:k] # filter only the k best results to return
        return best_fits


class ModelPicker: # create a singleton of the model below:
    jobs_similarity_class = None

    @staticmethod
    def get_instance():
        if not ModelPicker.jobs_similarity_class:
            ModelPicker.jobs_similarity_class = JobSimilarityClass()
            return ModelPicker.jobs_similarity_class
        else:
            return ModelPicker.jobs_similarity_class

