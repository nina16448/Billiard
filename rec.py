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

# The table you want to get data from
table = dynamodb.Table("BilliardTable")


# Convert DynamoDB Decimal type to float for json serialization
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)


while True:
    for id in [1, 2]:
        # Try to get the item from the table.
        response = table.get_item(Key={"Id": id})
        item = response.get("Item")

        # If the item exists, delete it and save the necessary data to a .json file.
        if item is not None:
            print(f"Item with ID {id} found:", item)
            table.delete_item(Key={"Id": id})

            # Remove 'Id' from the item
            del item["Id"]

            if id == 1:
                # Save the item to a .json file
                with open("HitParams.json", "w") as file:
                    json.dump(item, file, cls=DecimalEncoder)

                print("Data saved to HitParams.json")
                # print('Waiting for 5 seconds...')
                time.sleep(5)  # Wait for 5 seconds.

            if id == 2:
                with open("CueBallPosition.json", "w") as file:
                    json.dump(item, file, cls=DecimalEncoder)

                print("Position data saved to CueBallPosition.json")

                # print('Waiting for 3 seconds...')
                time.sleep(3)  # Wait for 3 seconds.

    # Wait for a while before trying again.
    print("Waiting...")
    time.sleep(0.1)  # Wait for 0.1 seconds.
