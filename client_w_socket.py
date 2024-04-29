import socket
import msgpack
import msgpack_numpy
import json
import torch
from torch import multiprocessing
import time
import os
import pickle
import argparse
import random
from typing import Dict, List, Tuple
import numpy as np
from dotenv import load_dotenv


parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--node-id", type=int, required=True, choices=range(0, 10))
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=65432, choices=range(0, 65536))
parser.add_argument("--db_postfix", type=str, default="")
ARGS = parser.parse_args()

load_dotenv(f"database{ARGS.db_postfix}/.env")

if os.getenv('MODEL') == 'SNN':
    import SNN as NN
elif os.getenv('MODEL') == 'ANN':
    import ANN as NN
else:
    raise Exception("Model type wrong!!")

NOISE_ABS_STD =  None if os.getenv('NOISE').split(',')[0] == '_' else float(os.getenv('NOISE').split(',')[0])
NOISE_PERCENTAGE_STD = None if os.getenv('NOISE').split(',')[1] == '_' else float(os.getenv('NOISE').split(',')[1])

from fedlearn import sha256_hash, set_parameters, get_parameters, add_percentage_gaussian_noise_to_model, add_constant_gaussian_noise_to_model


gpu_assignment = [int(x) for x in os.getenv('GPU_ASSIGNMENT').split(',')]

DEVICE = torch.device(f"cuda:{gpu_assignment[ARGS.node_id]}" if torch.cuda.is_available() else "cpu")
CLIENT_DATABASE_PATH = f"database{ARGS.db_postfix}/client_database_{ARGS.node_id}.json"
RESULT_CACHE_PATH = f"database{ARGS.db_postfix}/training_result_{ARGS.node_id}.pkl"


class PeripheralFL():
    def __init__(self, client_logs = []) -> None:
        self.local_model_result = None
        self.local_model_payload = None

        self.local_traing_process = None
        self.local_training_in_progress = False

        self.current_training_epoch = None if len(client_logs) == 0 else client_logs[-1]['epoch']
        self.client_logs = client_logs
        
        if os.path.exists(RESULT_CACHE_PATH):
            os.remove(RESULT_CACHE_PATH)


    def fit(
        self,  
        parameters: List[np.ndarray], 
        config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Load data
        trainloader, testloader = NN.load_client_data(ARGS.node_id)

        # Set model parameters, train model, return updated model parameters
        model = NN.load_model().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)

        set_parameters(model, parameters)

        NN.train(model, optimizer, trainloader, DEVICE, 1)
        loss, accuracy = NN.test(model, testloader, DEVICE)

        if NOISE_ABS_STD is not None:
            add_constant_gaussian_noise_to_model(model, DEVICE, NOISE_ABS_STD)
        if NOISE_PERCENTAGE_STD is not None:
            add_percentage_gaussian_noise_to_model(model, DEVICE, NOISE_PERCENTAGE_STD)

        trained_local_result = (get_parameters(model), len(trainloader.dataset))

        with open(RESULT_CACHE_PATH, 'wb') as file:
            pickle.dump((trained_local_result, loss, accuracy), file)

        # break

        del model, trainloader, testloader, trained_local_result, loss, accuracy

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return 
        # while True:
        #     try:
                
        #     except Exception as e:
        #         print(f"Training failed {e}")
        #         print(f"Sleeping for random seconds before retrying")
        #         time.sleep(random.randrange(5, 14))

                # with open(RESULT_CACHE_PATH, 'w') as file:
                #     file.write("FAILED")

    def is_running_local_training(self):
        return self.local_traing_process is not None and self.local_traing_process.is_alive()
    
    def discard_local_training(self):
        if self.is_running_local_training():
            self.local_traing_process.terminate()
            self.client_logs.append({
                'epoch': self.current_training_epoch,
                'status': 'TRAINING_TERMINATED'
            })

        self.local_model_result = None
        self.local_model_payload = None
        
        self.local_traing_process = None
        self.local_training_in_progress = False

        if os.path.exists(RESULT_CACHE_PATH):
            os.remove(RESULT_CACHE_PATH)

    
    def spawn_new_local_training(self, epoch, parameters):
        self.discard_local_training()
        
        self.local_traing_process = multiprocessing.Process(target=self.fit, args=(parameters, {}))
        # self.local_traing_process = threading.Thread(target=self.fit, args=(parameters, {}))

        if os.path.exists(RESULT_CACHE_PATH):
            os.remove(RESULT_CACHE_PATH)

        self.current_training_epoch = epoch
        self.local_traing_process.start()
        self.local_training_in_progress = True
    
    def load_local_training_result_if_done(self):
        if not os.path.exists(RESULT_CACHE_PATH) or self.is_running_local_training():
            return False
        
        with open(RESULT_CACHE_PATH, 'rb') as file:
            self.local_model_result, self.local_model_loss, self.local_model_accuracy = pickle.load(file)

        self.local_model_payload = msgpack.packb({
                'epoch': self.current_training_epoch,
                'params': self.local_model_result, 
                'accuracy': self.local_model_accuracy,
                'loss': self.local_model_loss,
            }, default=msgpack_numpy.encode)
        
        if len(self.client_logs) == 0 or self.client_logs[-1]['epoch'] != self.current_training_epoch \
        or self.client_logs[-1]['status'] != 'TRAINING_COMPLETED' \
        or sha256_hash(self.local_model_payload) != self.client_logs[-1]['local_model_payload_hash']:
            self.client_logs.append({
                'epoch': self.current_training_epoch,
                'status': 'TRAINING_COMPLETED',
                'loss': self.local_model_loss,
                'accuracy': self.local_model_accuracy, 
                'local_model_payload_size': len(self.local_model_payload),
                'local_model_payload_hash': sha256_hash(self.local_model_payload)
            })

        self.local_traing_process.join()
        self.local_training_in_progress = False

        print(json.dumps(self.client_logs[-1], indent=2))
        return True

class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.hand_shake_completed = False

        self.database = self.read_or_initiate_database()
        self.client_uid = self.database['client_uid']
        

        self.peripheral_fl = PeripheralFL(self.database['client_logs'])
        # self.sock.settimeout(5.0)

    def write_database(self, data):
        with open(CLIENT_DATABASE_PATH, 'w') as file:
            json.dump(data, file, indent=2)

    def read_or_initiate_database(self):
        if not os.path.isfile(CLIENT_DATABASE_PATH):
            self.write_database({
                'client_uid': None,
                'client_logs': []
            })

        with open(CLIENT_DATABASE_PATH, 'r') as file:
            data = json.load(file)
        
        return data
    
    def make_backup(self):
        self.write_database({
            'client_uid': self.client_uid,
            'client_logs': self.peripheral_fl.client_logs
        })

    def connect(self):
        try:
            self.sock.connect((self.host, self.port))
            print(f"Connected to server {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to server {self.host}:{self.port}: {e}")
            return
        
        self.listen_to_server()
        self.close()

    def handle_identity_prompt(self):
        if self.client_uid is None:
            self.send_to_server("NEW_CLIENT_REQUEST_ID")
        else:
            self.send_to_server(f"EXISTING_CLIENT_SUBMIT_ID:{self.client_uid}")

        return True

    def handle_new_identity_assigned(self, message):
        if self.client_uid is not None:
            print(f"Server attempts to assign new id on existing client: {self.client_uid}")
            return False
        else:
            self.client_uid = message.split(':')[1]
            print(f"Server assigned client_uid: {self.client_uid}. Connection established.")
            self.send_to_server(f"NEW_CLIENT_SUBMIT_ACK:{self.client_uid}")
            return True

    def handle_existing_client_ack(self, message):
        if self.client_uid != message.split(':')[1]:
            print(f"Server acknowledged wrong client id")
            return False
        else:
            print(f"Connection with server established")
            self.send_to_server(f"EXISTING_CLIENT_SUBMIT_ACK:{self.client_uid}")
            return True

    def handle_unrecognized_message(self, message):
        print(f"Unable to recognize message: {message}")

    def receive_data(self, data_size):
        received_data = b''
        while len(received_data) < data_size:
            try:
                more_data = self.sock.recv(data_size - len(received_data))
                if not more_data:
                    raise Exception("Server closed the connection unexpectedly.")
                received_data += more_data
            except socket.timeout:
                raise Exception("Timed out waiting for data from server.")
            
        return received_data
    
    def receive_global_model(self, global_model_payload_size):
        # Let server know client is ready for model params transfer
        self.send_to_server(f"CLIENT_INITIATE_GLOBAL_MODEL_RECEIVE:::{global_model_payload_size}")

        # Receive the model params data and unpack it
        global_model_payload = self.receive_data(global_model_payload_size)
        global_model_params = msgpack.unpackb(global_model_payload, object_hook=msgpack_numpy.decode)

        # Notify server about received model params
        self.send_to_server(f"CLIENT_CONFIRM_MODEL_RECEIVED:::{sha256_hash(global_model_payload)}")

        return global_model_params

    def local_params_transfer(self):
        self.transfer_data_to_server(self.peripheral_fl.local_model_payload)

        message = self.sock.recv(1024).decode('utf-8')
        print(f"Receiving from server: {message}")

        if not message.startswith("SERVER_CONFIRM_RECEIVED_MODEL") or message.split(':::')[1] != sha256_hash(self.peripheral_fl.local_model_payload):
            raise Exception("Not receiving the right signal confirmation from server for params transfer")
        else:
            print("Complete transfer local model params")

    def start_new_local_training(self, server_status):
        global_model_params = self.receive_global_model(server_status['global_model_size'])
        print(f"Starting new local training and cancel previous ones, epoch: {server_status}")
        self.peripheral_fl.spawn_new_local_training(server_status['current_training_epoch'], global_model_params)
        self.make_backup()

    def is_local_training_behind(self, server_status):
        return server_status['global_training_epoch'] != self.peripheral_fl.current_training_epoch
    
    def hand_shake(self):
        message = self.sock.recv(1024).decode('utf-8')
        print(f"Received from server: {message}")
        if not message == "SERVER_IDENTIFY_CLIENT" or not self.handle_identity_prompt():
            raise Exception(f"Hand shake failed at: {message}")
        
        message = self.sock.recv(1024).decode('utf-8')
        if message.startswith("SERVER_ASSIGN_NEW_CLIENT_ID"):
            if not self.handle_new_identity_assigned(message):
                raise Exception(f"Hand shake failed at: {message}")
        elif message.startswith("SERVER_ACK_EXISTING_CLIENT"):
            if not self.handle_existing_client_ack(message):
                raise Exception(f"Hand shake failed at: {message}")
        
        self.hand_shake_completed = True

    def frequent_status_check(self):
        # Craft client status message
        client_status = {
            'epoch': self.peripheral_fl.current_training_epoch
        }

        if self.peripheral_fl.local_training_in_progress:
            if self.peripheral_fl.load_local_training_result_if_done():
                self.make_backup()

        if self.peripheral_fl.local_training_in_progress:
            client_status['status'] = 'TRAINING_IN_PROGRESS'
        elif self.peripheral_fl.local_model_payload is None:
                client_status['status'] = 'NO_TRAINING_RESULT'
        else:
            client_status['status'] = 'TRAINING_COMPLETED'
            client_status['local_model_payload_size'] = self.peripheral_fl.client_logs[-1]['local_model_payload_size']

        # print(client_status)

        self.send_to_server(f"CLIENT_SEND_STATUS:::{json.dumps(client_status)}")

        message = self.sock.recv(1024).decode('utf-8')
        if not message.startswith("SERVER_SEND_STATUS"):
            print(f"Client status check not receiving correct response. Sleeping 5 sec and try again")
            time.sleep(5)
            return
        
        server_status = json.loads(message.split(':::')[1])

        if client_status['status'] == 'TRAINING_IN_PROGRESS':
            if server_status['command'] != 'KEEP_GOING':
                print("Something's wrong")
        elif client_status['status'] == 'TRAINING_COMPLETED':
            if server_status['command'] == 'STAY_PUT':
                pass
            elif server_status['command'] == 'SEND_LOCAL_MODEL':
                self.local_params_transfer()
            elif server_status['command'] == 'START_NEW_LOCAL_TRAINING':
                self.start_new_local_training(server_status)
        elif client_status['status'] == 'NO_TRAINING_RESULT':
            if server_status['command'] == 'STAY_PUT':
                pass
            elif server_status['command'] == 'START_NEW_LOCAL_TRAINING':
                self.start_new_local_training(server_status)
        else:
            raise Exception("Something's wrong with client status")

    def listen_to_server(self):
        while True:
            if not self.hand_shake_completed:
                self.hand_shake()
                self.make_backup()
            else:
                self.frequent_status_check()
            time.sleep(5)

    def send_to_server(self, message):
        # print(f"Sending to server: {message}")
        self.transfer_data_to_server(message.encode('utf-8'))


    def transfer_data_to_server(self, data):
        self.sock.sendall(data)

    def close(self):
        self.sock.close()
        print("Connection closed.")

if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver')
    
    while True:
        try:
            client = Client(ARGS.host, ARGS.port)
            client.connect()
        except Exception as e:
            del client
            print(f"Sleep 5 seconds, then start again")
            time.sleep(5)



    # peripheral_fl = PeripheralFL()

    # params = get_parameters(NN.load_model().to(DEVICE))
    # peripheral_fl.spawn_new_local_training(0, params)