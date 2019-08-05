from sagemaker.pytorch import PyTorch

# ref: https://aws.amazon.com/sagemaker/pricing/instance-types/
instance_type = 'ml.p3.2xlarge'
train_data_path = 's3://rnd-ocr/linus/2DOCR/glnet/data/'
output_path = 's3://rnd-ocr/linus/2DOCR/glnet/log/'
code_location = 's3://rnd-ocr/linus/2DOCR/glnet/code/'
role = "arn:aws:iam::533155507761:role/service-role/AmazonSageMaker-ExecutionRole-20190312T160681"
source_dir = "."
pytorch_estimator = PyTorch(entry_point='train_cinnamon.py',
                            source_dir=source_dir,
                            code_location=code_location,
                            output_path=output_path,
                            role=role,
                            train_instance_type=instance_type,
                            train_instance_count=1,
                            train_volume_size=200,
                            base_job_name= 'glnet-mode-3-linus',
                            train_max_run=5*86400,  # 86400s ~ 1day
                            framework_version='1.0.0',
                            py_version="py3",
                            hyperparameters={"config": "train_cinnamon_stage_2.ini"})
pytorch_estimator.fit({"train": train_data_path})
