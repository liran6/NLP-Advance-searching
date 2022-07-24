from flask import Flask, redirect, url_for, request, jsonify
from proccess_requests import ModelPicker

app = Flask(__name__)


@app.route('/get_jobs', methods=['POST'])
def get_jobs():
    try:
        #parsing data from post request
        data = request.get_json(force=True)
        data = data.get('data', [])
        if len(data) == 0:
            return {'error': "No data in request"}
        job_titles = data['job_titles']
        job_ids = data['job_ids']
        user_prompt = data['user_prompt']
        k = data['k']
        # call an instance of the processing model
        job_similarity_class = ModelPicker.get_instance()
        best_jobs = job_similarity_class.get_best_jobs(job_titles, job_ids, user_prompt, int(k))
        best_titles = []
        best_ids = []
        best_confidence = []
        #parse return data from model to response
        for item in best_jobs:
            best_confidence.append(str(item[0][0]))
            best_titles.append(item[1])
            best_ids.append(item[2])

        data_dict = {'best_titles': best_titles, 'best_ids': best_ids, 'best_confidence': best_confidence}
        response = dict(data=data_dict)
    except Exception as e:
        print(e)
        response = {'error': "Failed to handle request"}

    return response


if __name__ == '__main__':
    app.run(debug=True)