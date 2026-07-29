import joblib
import pandas as pd

def model_fn(model_dir):
    """
    Load the serialized model from the specified directory.
    This function is called by SageMaker once during container startup.

    Parameters:
    ----------
    model_dir : str
        The directory path where SageMaker stores the model artifacts.

    Returns:
    -------
    model : XGBClassifier
        The loaded Scikit-Learn/XGBoost model object.
    """
    # Load the serialized joblib model file from model directory
    model = joblib.load(f"{model_dir}/fraud_detection_model.pkl")
    return model

def input_fn(request_body, request_content_type):
    """
    Deserialize the incoming request body into an object suitable for inference.
    
    Parameters:
    ----------
    request_body : str or bytes
        The raw request payload sent to the endpoint.
    request_content_type : str
        The MIME type of the incoming payload (expects "text/csv").

    Returns:
    -------
    df : pandas.DataFrame
        DataFrame containing parsed feature inputs ready for the model.
    """
    # Verify the incoming content type is CSV
    if request_content_type == "text/csv":
        from io import StringIO
        # Parse the CSV string into a DataFrame without headers (matching training format)
        return pd.read_csv(StringIO(request_body), header=None)

    # Raise an error for unsupported content types
    raise ValueError("Unsupported content type")

def predict_fn(input_data, model):
    """
    Generate predictions using the loaded model and the parsed input data.

    Parameters:
    ----------
    input_data : pandas.DataFrame
        The deserialized feature input data.
    model : XGBClassifier
        The loaded model object returned by model_fn.

    Returns:
    -------
    predictions : numpy.ndarray
        Array containing predicted labels (e.g., [0, 1]).
    """
    # Perform prediction on the input features DataFrame
    return model.predict(input_data)

def output_fn(prediction, accept):
    """
    Serialize the output prediction array back into the response payload format.

    Parameters:
    ----------
    prediction : numpy.ndarray
        The prediction array returned by predict_fn.
    accept : str
        The MIME type requested by the client for the response.

    Returns:
    -------
    response_body : str
        A comma-separated string representation of the predictions.
    """
    # Return predictions formatted as a comma-separated string (e.g., "0,1,0")
    return ",".join(map(str, prediction))
