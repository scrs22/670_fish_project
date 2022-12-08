import datetime
from pathlib import Path
import boto3
import os
from botocore.exceptions import ClientError

bucket_name = "670-herring-dataset-unified"

target_folder = "./unified_fish_dataset"

s3r = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_FISHNET_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_FISHNET_ACCESS_KEY"),
)

s3c = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_FISHNET_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_FISHNET_ACCESS_KEY"),
)


progress = 0
start_data = datetime.datetime.now()
for fish_file in Path(target_folder).rglob("*.[tT][xX][tT]"):
    if fish_file.is_file():
        progress += 1
        target_file_name = os.path.relpath(fish_file, target_folder)
        s3r.upload_file(str(fish_file), bucket_name, target_file_name)

        if progress % 100 == 0:
            print(f"Uploaded {progress} Data Files.")
end_data = datetime.datetime.now()
print(f"Finished Data Uploading! Total Time {end_data - start_data}")

progress = 0
start_img = datetime.datetime.now()
for fish_file in Path(target_folder).rglob("*.[pP][nN][gG]"):
    if fish_file.is_file():
        progress += 1
        target_file_name = os.path.relpath(fish_file, target_folder)
        try:
            s3r.Object(bucket_name, target_file_name).load()
        except ClientError as e:
            s3c.upload_file(str(fish_file), bucket_name, target_file_name)
        if progress % 50 == 0:
            print(f"Uploaded {progress} Image Files.")
end_img = datetime.datetime.now()
print(f"Finished Image Uploading! Total Time {end_img - start_img}")
