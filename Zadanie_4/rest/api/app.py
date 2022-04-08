from flask import Blueprint, request, jsonify, Flask
from model import predict

prediction_app = Blueprint('prediction_app', __name__)

@prediction_app.route('/status', methods=['GET'])
def get_state():
  return jsonify({'status': 'online'})

@prediction_app.route('/version', methods=['GET'])
def get_version():
  return jsonify({'version': predict.__version__})

@prediction_app.route('/predict', methods=['POST'])
def post_predict():
  result = predict.run(request.get_json(force=True))
  return jsonify(result)


flask = Flask('app')
flask.register_blueprint(prediction_app)
flask.run()
