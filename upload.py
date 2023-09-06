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


class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)


table = dynamodb.Table("BilliardTable")

# 要監控的目錄
directory_to_watch = "./"  # 替換為你要監控的目錄
t_reset = 0
while True:
    t = time.time()
    # 檢查 'Hit.json' 是否存在於目錄中

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

    elif "Hit.json" in os.listdir(directory_to_watch):
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

    # id = 5

    if t - t_reset > 3:
        print("Wait...")

        for id in [3, 4, 5, 6]:
            response = table.get_item(Key={"Id": id})
            item = response.get("Item")

            if item is not None:
                t_reset = t
                print(f"Item with ID {id} found:", item)
                table.delete_item(Key={"Id": id})
                del item["Id"]

                if id == 3:
                    with open("./unity/Player1_Data/HitParams.json", "w") as file:
                        json.dump(item, file, cls=DecimalEncoder)

                    print("Data saved to HitParams.json")

                if id == 4:
                    with open("./unity/Player1_Data/CueBallPosition.json", "w") as file:
                        json.dump(item, file, cls=DecimalEncoder)

                    print("Position data saved to CueBallPosition.json")
                if id == 5:
                    with open("reset.json", "w") as file:
                        json.dump(item, file, cls=DecimalEncoder)
                    print("reset")
                    t_reset = t_reset - 2
                if id == 6:
                    with open("turn.json", "w") as file:
                        json.dump(item, file, cls=DecimalEncoder)
                    print("turn")
                    t_reset = t_reset - 2

    time.sleep(0.1)  # Wait for 0.1 seconds.
