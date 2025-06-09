import os

import aioboto3

from botocore.client import Config

from typing import Any

from config.s3 import S3Settings
from storage.s3.s3_storage_cons import Bucket
from storage.protocol.protocol import Storage, StorageResult


class S3Result(StorageResult):
    data: Any
    message: str | None = None


class BaseStorage(Storage):
    def __init__(self, s3_settings: S3Settings):
        self.bucket_name = Bucket.MAIN.value
        self.global_bucket_name = Bucket.MAIN.value
        self.global_endpoint_url = s3_settings.global_endpoint_url
        self.s3_path = s3_settings.path
        self.s3_conf = {
            "service_name": s3_settings.service_name,
            "endpoint_url": s3_settings.endpoint_url,
            "region_name": s3_settings.region_name,
            "aws_access_key_id": s3_settings.aws_access_key_id,
            "aws_secret_access_key": s3_settings.aws_secret_access_key,
            "config": Config(
                signature_version="s3v4",
                s3={"addressing_style": s3_settings.addressing_style},
            ),
        }
        self.session = aioboto3.Session()

    async def send_to_storage(self, file_contents: bytes, file_name: str) -> S3Result:
        async with self.session.client(**self.s3_conf) as s3:
            try:
                extra_args = {
                    "ACL": "public-read",
                    "Metadata": {"CacheControl": "max-age=31536000"},
                }
                if file_name.endswith("svg"):
                    extra_args["ContentType"] = "image/svg+xml"

                await s3.upload_fileobj(
                    file_contents,
                    self.bucket_name,
                    f"{self.s3_path}/{file_name}",
                    ExtraArgs=extra_args,
                )

                object_url = f"{self.global_endpoint_url}/{self.global_bucket_name}/{self.s3_path}/{file_name}"
                return S3Result(
                    data=object_url,
                    message="Файл успешно загружен",
                )
            except s3.exceptions.ClientError as e:
                return S3Result(
                    data=None,
                    message=f"Ошибка при загрузке файла в хранилище: {e}",
                )

    async def get_by_name(self, file_name: str) -> S3Result:
        async with self.session.client(**self.s3_conf) as s3:
            try:
                file_key = f"{self.s3_path}/{file_name}"
                await s3.head_object(Bucket=self.bucket_name, Key=file_key)
                object_url = f"{self.global_endpoint_url}/{self.global_bucket_name}/{file_key}"
                result = S3Result(
                    data=object_url,
                    message=f"File {file_name} has been found in the storage.",
                )
                return result
            except s3.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return S3Result(
                        data=None,
                        message=f"File '{file_name}' does not exist in the storage.",
                    )
                else:
                    raise

    async def delete_by_name(self, file_name: str) -> S3Result:
        async with self.session.client(**self.s3_conf) as s3:
            try:
                file_key = f"{self.s3_path}/{file_name}"
                await s3.delete_object(Bucket=self.bucket_name, Key=file_key)
                return S3Result(
                    data=None,
                    message=f"File '{file_name}' has been deleted from the storage.",
                )
            except s3.exceptions.ClientError as e:
                return S3Result(
                    data=None,
                    message=f"Error deleting file '{file_name}' from the storage: {e}",
                )

    async def list_all(self) -> S3Result:
        async with self.session.client(**self.s3_conf) as s3:
            try:
                temp = []
                objects = await s3.list_objects(Bucket=self.bucket_name)
                for key in objects.get("Contents", []):
                    temp.append(key["Key"])
                result = S3Result(data=temp, message="List all objects in the storage.")
                return result
            except s3.exceptions.ClientError as e:
                result = S3Result(
                    data=None,
                    message=f"Error listing objects in the storage: {e}",
                )
                return result

    async def download_all_from_storage(self) -> S3Result:
        download_folder = os.path.join(os.getcwd(), "downloads")
        os.makedirs(download_folder, exist_ok=True)

        async with self.session.client(**self.s3_conf) as s3:
            try:
                objects = await s3.list_objects(Bucket=self.bucket_name)
                for key in objects.get("Contents", []):
                    file_key = key["Key"]
                    downloaded_file_name = os.path.join(download_folder, file_key.split("/")[-1])
                    await s3.download_file(self.bucket_name, file_key, downloaded_file_name)

                result = S3Result(
                    data=None,
                    message=f"All files downloaded to {download_folder}",
                )
                return result

            except s3.exceptions.ClientError as e:
                result = S3Result(
                    data=None,
                    message=f"Error downloading files from storage: {e}",
                )
                return result
