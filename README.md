# Federated Learning Framework

This README provides instructions on how to set up and run a federated learning framework using basic socket-based server and client Python scripts.

## Requirements

Ensure you have Python 3 installed on your system. You can download Python from the official website: [Python.org](https://www.python.org/downloads/).

## Setup

Clone or download this repository to your local machine.

## Running the Server

To run the server component of the federated learning framework, use the following command from the terminal:

```bash
python3 serve_w_socket.py --host 0.0.0.0 --db_postfix _test

## Running the Client

```bash
python3 client_w_socket.py --host <host_ip> --db_postfix _test
