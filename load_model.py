import mlflow
import numpy as np
import pandas as pd

# Set tht MLFlow server uri
uri = "http://127.0.0.1:6001"
mlflow.set_tracking_uri(uri=uri)

# Provide model path/url
# logged_model = "mlflow-artifacts:/626345054421746950/f4f0511850b24621973a6502a582716b/artifacts/model_artifacts"
logged_model = "mlflow-artifacts:/626345054421746950/8252502785d54d2a8522102faa45550c/artifacts/model_artifacts"

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# Input a random datapoint
np.random.seed(42)
data = np.random.rand(1, 64)

# Predict the output for the data. You might need to use a pandas DataFrame due to a constraint from MLFlow.
data_df = pd.DataFrame(data)
prediction = loaded_model.predict(data_df)

# Print out prediction result
print(prediction)
