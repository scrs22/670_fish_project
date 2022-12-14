import datetime

import boto3
import os

bucket_name = "herring-ds4cg"

herring_prefix = "Herring/"
non_herring_prefix = "Non-herring/"

target_folder = "original_fish_dataset/"

s3r = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_FISHNET_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_FISHNET_ACCESS_KEY"),
)
bucket = s3r.Bucket(bucket_name)

progress = 0
start = datetime.datetime.now()
for object_summary in bucket.objects.filter(Prefix=non_herring_prefix):
    progress += 1
    target_download_folder = os.path.join(
        target_folder, os.path.dirname(object_summary.key)
    )
    target_download_file_name = os.path.join(
        target_download_folder, os.path.basename(object_summary.key)
    )

    os.makedirs(target_download_folder, exist_ok=True)
    bucket.download_file(object_summary.key, target_download_file_name)
    if progress % 100 == 0:
        print(f"Downloaded {progress} Images.")
end = datetime.datetime.now()
print(f"Finished Downloading! Total Time {end-start}")
