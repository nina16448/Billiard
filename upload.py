import boto3
import os
import json
from decimal import Decimal
import time

# 設置 AWS 憑證
aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
region_name = os.environ["AWS_REGION"]  # 你的 AWS 區域代碼

dynamodb = boto3.resource(
    "dynamodb",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name,
)

table = dynamodb.Table("BilliardTable")

# 要監控的目錄
directory_to_watch = "./"  # 替換為你要監控的目錄

while True:
    # 檢查 'Hit.json' 是否存在於目錄中
    if "Hit.json" in os.listdir(directory_to_watch):
        print("get data Hit.json")
        # 如果存在，讀取該檔案
        with open(os.path.join(directory_to_watch, "Hit.json"), "r") as f:
            data = json.load(f)

        # 讀取後刪除該檔案
        os.remove(os.path.join(directory_to_watch, "Hit.json"))

        # 確保主鍵值設定為 1
        data["Id"] = 1

        # 將 'data' 中的任何浮點數轉換為 Decimals
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, float):
                        data[key][subkey] = Decimal(str(subvalue))
            elif isinstance(value, float):
                data[key] = Decimal(str(value))

        # 將資料推送至 DynamoDB 表格
        response = table.put_item(Item=data)

        print("Hit.json upload success!")

    if "Position.json" in os.listdir(directory_to_watch):
        print("get data Position.json")
        # 如果存在，讀取該檔案
        with open(os.path.join(directory_to_watch, "Position.json"), "r") as f:
            data = json.load(f)

        # 讀取後刪除該檔案
        os.remove(os.path.join(directory_to_watch, "Position.json"))

        # 確保主鍵值設定為 1
        data["Id"] = 2

        # 將 'data' 中的任何浮點數轉換為 Decimals
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, float):
                        data[key][subkey] = Decimal(str(subvalue))
            elif isinstance(value, float):
                data[key] = Decimal(str(value))

        # 將資料推送至 DynamoDB 表格
        response = table.put_item(Item=data)

        print("Position.json upload success!")
