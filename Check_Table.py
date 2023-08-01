import boto3
import time
import json
from decimal import Decimal

# 設置 AWS 憑證
aws_access_key_id = "AKIAQO7NA2QJS6HPVLFW"
aws_secret_access_key = "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"
region_name = "ap-northeast-1"  # 你的 AWS 區域代碼

dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# The table you want to get data from
table = dynamodb.Table('BilliardTable')

# Convert DynamoDB Decimal type to float for json serialization
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)

while True:
    for id in [1, 2]:
        # Try to get the item from the table.
        response = table.get_item(Key={'Id': id})
        item = response.get('Item')

        # If the item exists, delete it and save the necessary data to a .json file.
        if item is not None:
            print(f'Item with ID {id} found:', item)
            table.delete_item(Key={'Id': id})

            # Remove 'Id' from the item
            del item['Id']

            if id == 1:
                # Save the item to a .json file
                with open('HitParams.json', 'w') as file:
                    json.dump(item, file, cls=DecimalEncoder)

                print("Data saved to HitParams.json")

            if id == 2:
                with open('CueBallPosition.json', 'w') as file:
                    json.dump(item, file, cls=DecimalEncoder)

                print("Position data saved to CueBallPosition.json")

    # Wait for a while before trying again.
    print('Waiting...')
    time.sleep(0.01)  # Wait for 0.01 seconds.
