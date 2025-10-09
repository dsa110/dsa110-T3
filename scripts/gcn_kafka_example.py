#! /usr/bin/env

from gcn_kafka import Consumer, Producer
import argparse
import json
import sys
import os

if __name__ == "__main__":
    # Parse command line arguments to determine if running as consumer or producer and environment
    if len(sys.argv) < 2:
        print("Usage: python example.py -c (for consumer) or -p (for producer) [-e prod|test]")
        sys.exit(1)
    else:
        parser = argparse.ArgumentParser(description="Run as consumer or producer.")
        parser.add_argument('-m', '--mode', choices=['consumer', 'producer'], help='Run in consumer or producer mode')
        parser.add_argument('-e', '--env', choices=['prod', 'test'], default='test', help='Environment: prod or test (default: test)')
        parser.add_argument('-t', '--topic', default='gcn.notices.dsa110.frb', help='Kafka topic to use')
        args, unknown = parser.parse_known_args()

if args.env == 'test':
    domain = 'test.gcn.nasa.gov'
    if args.mode == "consumer":
        client_id = os.environ.get('GCN_ID_CON_TEST', '')
        client_secret = os.environ.get('GCN_SECRET_CON_TEST', '')
    if args.mode == "producer":
        client_id = os.environ.get('GCN_ID_PRO_TEST', '')
        client_secret = os.environ.get('GCN_SECRET_PRO_TEST', '')
else:
    domain = 'gcn.nasa.gov'
    print("Still need to get credentials for production mode")
    sys.exit(1)

if args.mode == "producer":
    print(client_id, domain, args.topic)
    # Connect as a producer (client "dsa110")
    producer = Producer(client_id=client_id, client_secret=client_secret, domain=domain)

    # JSON data converted to byte string format
    data = json.dumps({
        '$schema': 'https://gcn.nasa.gov/schema/v6.0.0/gcn/notices/mission/SchemaName.schema.json',
        'key': 'value'
    }).encode("utf-8")

    producer.produce(args.topic, data)
    producer.flush()
elif args.mode == 'consumer':
    # Connect as a consumer
    config = {'auto.offset.reset': 'earliest'}
    consumer = Consumer(client_id=client_id, client_secret=client_secret, domain=domain, config=config)

    # Subscribe to topics and receive alerts
    consumer.subscribe([args.topic])

    while True:
        for message in consumer.consume(timeout=1):
            if message.error():
                print(message.error())
                continue
            # Print the topic and message ID
            print(f'topic={message.topic()}, offset={message.offset()}')
            value = message.value()
            print(value)
