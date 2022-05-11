from sagemaker.pytorch import PyTorch
import sagemaker

#id & keys has been hidden
aws_access_key_id = ""
aws_secret_access_key = ""
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
role_name = role.split(["/"][-1])
print(f"The Amazon Resource Name (ARN) of the role used for this demo is: {role}")
print(f"The name of the role used for this demo is: {role_name[-1]}")

estimator = PyTorch(
    base_job_name="distributed_work",
    source_dir="code",
    entry_point="run_distributed.py",
    role=role,
    framework_version="1.8.1",
    py_version="py36",
    # For training with multinode distributed training, set this count. Example: 2
    instance_count=1,
    # For training with p3dn instance use - ml.p3dn.24xlarge, with p4dn instance use - ml.p4d.24xlarge
    instance_type="ml.p3.16xlarge",
    sagemaker_session=sagemaker_session,
    # Training using SMDataParallel Distributed Training Framework
    distribution={"smdistributed": {"dataparallel": {"enabled": True}}},
    debugger_hook_config=False,
)
estimator.fit("s3://bucket/path/to/training/data")
model_data = estimator.model_data
print("Storing {} as model_data".format(model_data))
# %store model_data