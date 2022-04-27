import copy
import time

from flask import Flask, jsonify, abort, make_response, request
from rq import Queue, get_current_job
from redis import Redis
import logging
import numpy as np

IRIS = ('setosa', 'versicolor', 'virginica')

redis_conn = Redis(host='localhost', port=6379)
queue = Queue('rest_api', connection=redis_conn, default_timeout=1200)

logging.basicConfig(filename='logs/logs.log', level=logging.DEBUG)

app = Flask(__name__)


def launch_task(sepal_length, sepal_width, petal_length, petal_width, api):
    logging.info('sepal_length, sepal_width, petal_length, petal_width, api')
    if api == 'v1.0':
        prediction = get_pred(sepal_length, sepal_width, petal_length, petal_width)
        iris_name = IRIS[prediction]
        res_dict = {'result': iris_name}
        logging.info(res_dict)
    else:
        res_dict = {'error': 'API doesnt exist'}
    return res_dict


def get_response(res_dict, status=200):
    return make_response(jsonify(res_dict), status)


def get_job_response(job_id):
    return get_response({'job_id': job_id})


@app.route('/iris/api/v1.0/getpred', methods=['GET'])
def get_task():
    job_id = request.args.get('job_id')
    job = queue.enqueue('rest_api.launch_task',
                        request.args.get('sepal_length'), request.args.get('sepal_width'),
                        request.args.get('petal_length'), request.args.get('petal_width'),
                        'v1.0',
                        result_ttl=60 * 60 * 24,
                        job_id=job_id)

    return get_job_response(job.get_id())


def get_pred(sepal_length, sepal_width, petal_length, petal_width):
    time.sleep(4)
    w = np.random.randn(5, 3)
    x = np.array([float(sepal_length), float(sepal_width), float(petal_length), float(petal_width), 1])
    return np.argmax(x @ w)


def get_process_response(code, process_status, status=200):
    return get_response({'code': code, 'status': process_status}, status)


@app.route('/iris/api/v1.0/status/<job_id>')
def status(job_id):
    job = queue.fetch_job(job_id)
    if job is None:
        return get_process_response('NOT_FOUND', 'error', 404)
    if job.is_failed:
        return get_process_response('INTERNAL SERVER ERROR', 'error', 500)
    if job.is_finished:
        iris_name = copy.deepcopy(job.result)['result']
        result = {'result': iris_name}
        return get_response(result)
    return get_process_response('NOT_READY', 'running', 202)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'code': 'PAGE_NOT_FOUND'}), 404)


@app.errorhandler(500)
def server_error(error):
    return make_response(jsonify({'code': 'INTERNAL_SERVER_ERROR'}), 500)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
