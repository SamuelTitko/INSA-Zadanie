import joblib
import logging

logger = logging.getLogger(__name__)

def get_version(file_name):
  with open(f'{file_name}', 'r') as file:
    version = file.read()
  return version

def save_pipeline(path, file_name, version, pipeline, fmt='.pkl'):
  pipeline_name = f'{file_name}-{version}{fmt}'
  joblib.dump(pipeline, path / pipeline_name)
  logger.info(f'Pipeline {pipeline_name} saved.')

def load_pipeline(path, file_name, version, fmt='.pkl'):
  pipeline_name = f'{file_name}-{version}{fmt}'
  pipeline = joblib.load(path / pipeline_name)
  logger.info(f'Pipeline {pipeline_name} loaded.')
  return pipeline
