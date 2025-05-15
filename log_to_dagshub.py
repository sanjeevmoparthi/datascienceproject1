import dagshub
dagshub.init(repo_owner='sanjeevmoparthi', repo_name='datascienceproject1', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
  mlflow.autolog()