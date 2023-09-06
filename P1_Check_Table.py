import boto3
import time
import json
from decimal import Decimal

# Initialize a DynamoDB client
dynamodb = boto3.resource('dynamodb')

# The table you want to get data from
table = dynamodb.Table('BilliardTable')

# Convert DynamoDB Decimal type to float for json serialization
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)

while True:
    for id in [3, 4, 5]:
        # Try to get the item from the table.
        response = table.get_item(Key={'Id': id})
        item = response.get('Item')

        # If the item exists, delete it and save the necessary data to a .json file.
        if item is not None:
            print(f'Item with ID {id} found:', item)
            table.delete_item(Key={'Id': id})

            # Remove 'Id' from the item
            del item['Id']

            if id == 3:
                # Save the item to a .json file
                with open('HitParams.json', 'w') as file:
                    json.dump(item, file, cls=DecimalEncoder)

                print("Data saved to HitParams.json")
                #print('Waiting for 5 seconds...')
                time.sleep(5)  # Wait for 5 seconds.

            if id == 4:
                with open('CueBallPosition.json', 'w') as file:
                    json.dump(item, file, cls=DecimalEncoder)

                print("Position data saved to CueBallPosition.json")
                
                #print('Waiting for 3 seconds...')
                time.sleep(0.1)  # Wait for 0.1 seconds.
            if id == 5:
                print("Reset message receive")
                time.sleep(1)  # Wait for 3 seconds.


    # Wait for a while before trying again.
    print('Waiting...')
    time.sleep(0.1)  # Wait for 0.1 seconds.
