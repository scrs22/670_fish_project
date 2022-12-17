import datetime
from pathlib import Path
import boto3
import os
from botocore.exceptions import ClientError

bucket_name = "670-dataset-uncropped-denoise-paul"

cur_path = os.getcwd()
target_folder = './dataset_cropped_denoise'

s3r = boto3.resource(
    "s3",
    aws_access_key_id="AKIARKW6BIIZBIOXAOXA",
    aws_secret_access_key="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",
)

s3c = boto3.client(
    "s3",
    aws_access_key_id="AKIARKW6BIIZBIOXAOXA",
    aws_secret_access_key="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",
)


progress = 0
start_data = datetime.datetime.now()
for fish_file in Path(target_folder).rglob("*.[tT][xX][tT]"):
    if fish_file.is_file():
        progress += 1
        target_file_name = Path(os.path.relpath(fish_file, target_folder)).as_posix()
        s3c.upload_file(str(fish_file), bucket_name, target_file_name)

        if progress % 100 == 0:
            print(f"Uploaded {progress} Data Files.")
end_data = datetime.datetime.now()
print(f"Finished Data Uploading! Total Time {end_data - start_data}")

progress = 0
start_img = datetime.datetime.now()
for fish_file in Path(target_folder).rglob("*.[pP][nN][gG]"):
    if fish_file.is_file():
        progress += 1
        target_file_name = Path(os.path.relpath(fish_file, target_folder)).as_posix()
        try:
            s3r.Object(bucket_name, target_file_name).load()
        except ClientError as e:
            s3c.upload_file(str(fish_file), bucket_name, target_file_name)
        if progress % 50 == 0:
            print(f"Uploaded {progress} Image Files.")
end_img = datetime.datetime.now()
print(f"Finished Image Uploading! Total Time {end_img - start_img}")
