from gcn_kafka import Consumer, Producer
import argparse
import json
import sys
import os

# Connect as a consumer (client "dsa110")
# Warning: don't share the client secret with others.
client_id_test = os.environ.get('GCN_ID_TEST', '')
client_secret_test = os.environ.get('GCN_SECRET_TEST', '')
client_id_prod = os.environ.get('GCN_ID', '')
client_secret_prod = os.environ.get('GCN_SECRET', '')

if __name__ == "__main__":
    # Parse command line arguments to determine if running as consumer or producer and environment
    if len(sys.argv) < 2:
        print("Usage: python example.py -c (for consumer) or -p (for producer) [-e prod|test]")
        sys.exit(1)
    else:
        parser = argparse.ArgumentParser(description="Run as consumer or producer.")
        parser.add_argument('-c', '--asconsumer', action='store_true', help='Run as consumer')
        parser.add_argument('-p', '--asproducer', action='store_true', help='Run as producer')
        parser.add_argument('-e', '--env', choices=['prod', 'test'], default='test', help='Environment: prod or test (default: test)')
        args, unknown = parser.parse_known_args()

        if not args.asconsumer and not args.asproducer:
            print("Please specify either -c (consumer) or -p (producer).")
            sys.exit(1)
if args.env == 'test':
    client_id = client_id_test
    client_secret = client_secret_test
else:
    client_id = client_id_prod
    client_secret = client_secret_prod

if args.asproducer:
    # Connect as a producer (client "dsa110")
    producer = Producer(client_id=client_id, client_secret=client_secret)

    # Produce a message to the topic
    topic = 'gcn.notices.dsa110'

    # JSON data converted to byte string format
    data = json.dumps({
        '$schema': 'https://gcn.nasa.gov/schema/vX.Y.Z/gcn/notices/mission/SchemaName.schema.json',
        'key': 'value'
    }).encode()
    producer.produce(topic, data)
    producer.flush()

if args.asconsumer:
    # Connect as a consumer
    consumer = Consumer(client_id=client_id, client_secret=client_secret)

    # Subscribe to topics and receive alerts
    consumer.subscribe(['igwn.gwalert'])
    while True:
        for message in consumer.consume(timeout=1):
            if message.error():
                print(message.error())
                continue
            # Print the topic and message ID
            print(f'topic={message.topic()}, offset={message.offset()}')
            value = message.value()
            print(value)
