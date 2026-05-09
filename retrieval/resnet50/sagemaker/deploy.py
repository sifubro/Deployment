import boto3
import sagemaker
from sagemaker.model import Model

# run it from (base) anaconda not the torch

session = sagemaker.Session()
sm_client = boto3.client("sagemaker", region_name="eu-west-1")

#ACCOUNT_ID = 586917955410
role = "arn:aws:iam::586917955410:role/SageMakerExecutionRole"  # create this in IAM
image_uri = "586917955410.dkr.ecr.eu-west-1.amazonaws.com/depop-embedder:latest"


# Clean up any leftover configs/endpoints from previous attempts
for name in ["depop-embedder-endpoint"]:
    try:
        sm_client.delete_endpoint(EndpointName=name)
        print(f"Deleted endpoint: {name}")
    except sm_client.exceptions.ClientError:
        pass
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=name)
        print(f"Deleted endpoint config: {name}")
    except Exception:
        pass

model = Model(
    image_uri=image_uri,
    role=role,
    sagemaker_session=session,
    name="depop-embedder",
)

try:
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.t2.medium", # ml.t2.medium, ml.m5.large CPU; use ml.g4dn.xlarge for GPU
        endpoint_name="depop-embedder-endpoint",
    )
finally:
    predictor.delete_endpoint()
    print("Endpoint deleted.")