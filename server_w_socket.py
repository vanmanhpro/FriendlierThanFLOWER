import socket
import threading
import uuid
import msgpack
import msgpack_numpy
from torch import multiprocessing
import time
import torch
import json
import os
import pickle
import argparse
from dotenv import load_dotenv

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=65432, choices=range(0, 65536))
parser.add_argument("--db_postfix", type=str, default="")

ARGS = parser.parse_args()

# print(ARGS)

load_dotenv(f"database{ARGS.db_postfix}/.env")

if os.getenv('MODEL') == 'SNN':
    import SNN as NN
elif os.getenv('MODEL') == 'ANN':
    import ANN as NN

NOISE_ABS_STD =  None if os.getenv('NOISE').split(',')[0] == '_' else float(os.getenv('NOISE').split(',')[0])
NOISE_PERCENTAGE_STD = None if os.getenv('NOISE').split(',')[1] == '_' else float(os.getenv('NOISE').split(',')[1])
MIN_FIT_CLIENT = int(os.getenv('MIN_FIT_CLIENT'))

from fedlearn import sha256_hash, fedavg_aggregate, set_parameters, get_parameters, add_percentage_gaussian_noise_to_model, add_constant_gaussian_noise_to_model

DEVICE = torch.device(f"cuda:{os.getenv('SERVER_GPU_ASSIGNMENT')}" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")

DATA_PATH = os.getenv('DATA_PATH')
SERVER_DATABASE_PATH = f"database{ARGS.db_postfix}/server_database.pkl"
CONNECTION_DATABASE_PATH = f"database{ARGS.db_postfix}/connection_database.json"
SERVER_LOG_DATABASE_PATH = f"database{ARGS.db_postfix}/server_log_database.json"

TEMP_GLOBAL_MODEL_PATH = f"database{ARGS.db_postfix}/global_model_temp.pkl"
PERMANENT_GLOBAL_MODEL_PATH = f"database{ARGS.db_postfix}/global_model_permanent.pkl"

def write_global_log(global_model_epoch, global_loss, global_accuracy, epoch_start_time):
    if not os.path.isfile(SERVER_LOG_DATABASE_PATH):
        data = []
        with open(SERVER_LOG_DATABASE_PATH, 'w') as file:
            json.dump([], file)
    else:
        with open(SERVER_LOG_DATABASE_PATH, 'r') as file:
            data = json.load(file)

    data.append({
        'global_model_epoch': global_model_epoch,
        'global_loss': global_loss,
        'global_accuracy': global_accuracy,
        'training_time': time.time() - epoch_start_time
    })

    with open(SERVER_LOG_DATABASE_PATH, 'w') as file:
        json.dump(data, file, indent=2)


def write_global_model(file_path, global_model_epoch, global_model_params, global_loss, global_accuracy):
    global_model_params_payload = msgpack.packb(global_model_params, default=msgpack_numpy.encode)
    global_model_size = len(global_model_params_payload)

    with open(file_path, 'wb') as file:
        pickle.dump({
            'global_model_epoch': global_model_epoch,
            'global_model_params': global_model_params,
            'global_model_params_payload': global_model_params_payload,
            'global_model_size': global_model_size,
            'global_loss': global_loss,
            'global_accuracy': global_accuracy,
        }, file)

def read_global_model(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    return data

def read_or_initialize_global_model():
    if not os.path.isfile(PERMANENT_GLOBAL_MODEL_PATH):
        write_global_model(
            PERMANENT_GLOBAL_MODEL_PATH, -1, get_parameters(NN.load_model(CPU_DEVICE).to(CPU_DEVICE)), None, None
        )
    
    return read_global_model(PERMANENT_GLOBAL_MODEL_PATH)

def evaluate(global_model):
    test_loader = NN.load_test_data()
    
    loss, accuracy = NN.test(global_model, test_loader, DEVICE)

    return loss, accuracy

def centralized_aggregation(current_training_epoch, client_model_record):
    while True:
        try:
            client_results = [client_model_record[uid] for uid in client_model_record.keys()]
            global_model_params = fedavg_aggregate(client_results)

            global_model = NN.load_model().to(DEVICE)
            set_parameters(global_model, global_model_params)
            global_loss, global_accuracy = evaluate(global_model)

            if NOISE_ABS_STD is not None:
                add_constant_gaussian_noise_to_model(global_model, DEVICE, NOISE_ABS_STD)
            if NOISE_PERCENTAGE_STD is not None:
                add_percentage_gaussian_noise_to_model(global_model, DEVICE, NOISE_PERCENTAGE_STD)

            print(f"Aggregated result: Device: {DEVICE}, Training epoch {current_training_epoch}, Loss {global_loss}, Acc {global_accuracy}")

            write_global_model(TEMP_GLOBAL_MODEL_PATH, current_training_epoch, global_model_params, global_loss, global_accuracy)
            return
        except Exception as e:
            print(f"Error in aggregation: {e}")
            time.sleep(5)

class CentralizeFL():
    def __init__(self) -> None:

        gm_data = read_or_initialize_global_model()
        self.populate_global_model(gm_data)
        self.global_aggregation_process = None
        self.global_model_lock = threading.Lock()

        self.current_training_epoch = self.global_model_epoch + 1
        self.client_model_record = {}
        self.client_result_lock = threading.Lock()

        self.aggregation_running = False

        self.min_fit_clients = MIN_FIT_CLIENT

        self.epoch_start_time = time.time()

    def populate_global_model(self, gm_data=None):
        self.global_model_epoch = gm_data['global_model_epoch']
        self.global_model_params = gm_data['global_model_params']
        self.global_model_params_payload = gm_data['global_model_params_payload']
        self.global_model_size = gm_data['global_model_size']
        self.global_accuracy = gm_data['global_accuracy']
        self.global_loss = gm_data['global_loss']

    def start_aggregation_if_suffient_result(self):
        print(f"Checking model aggregation condition: {self.current_training_epoch, list(self.client_model_record.keys())}")
        if len(self.client_model_record.keys()) >= self.min_fit_clients:
            if self.is_aggregation_running():
                raise Exception("A global aggregation process is already running. Something is wrong with the code")
            
            print(f"Starting centralized model aggregation.")
            self.global_aggregation_process = multiprocessing.Process(target=centralized_aggregation, args=(self.current_training_epoch, self.client_model_record,))

            self.global_aggregation_process.start()

            # Delete existing global model
            self.global_model_epoch = None
            self.global_model_params = None 
            self.global_model_params_payload = None
            self.global_model_size = None
            self.global_accuracy = None
            self.aggregation_running = True
            if os.path.isfile(TEMP_GLOBAL_MODEL_PATH):
                # Delete the file
                os.remove(TEMP_GLOBAL_MODEL_PATH)


    def receive_client_result(self, client_uid, client_model):
        with self.client_result_lock:
            if client_uid not in self.client_model_record:
                print(f"Epoch {self.current_training_epoch}: Model {client_uid} added to {list(self.client_model_record.keys())}")
                self.client_model_record[client_uid] = client_model['params']
                self.start_aggregation_if_suffient_result()

    def is_aggregation_running(self):
        return (self.global_aggregation_process is not None and self.global_aggregation_process.is_alive())

    def get_aggregated_model_if_havent(self):
        with self.client_result_lock:
            if self.global_model_params is None and os.path.exists(TEMP_GLOBAL_MODEL_PATH):
                gm_data = read_global_model(TEMP_GLOBAL_MODEL_PATH)
                self.populate_global_model(gm_data)

                write_global_log(self.global_model_epoch, self.global_loss, self.global_accuracy, self.epoch_start_time)

                self.epoch_start_time = time.time()

                if os.path.isfile(PERMANENT_GLOBAL_MODEL_PATH):
                    # Delete the file
                    os.remove(PERMANENT_GLOBAL_MODEL_PATH)
                os.rename(TEMP_GLOBAL_MODEL_PATH, PERMANENT_GLOBAL_MODEL_PATH)
                
                self.aggregation_running = False
                self.current_training_epoch = self.global_model_epoch + 1
                self.client_model_record = {}
                print(f"AFTER get_aggregated_model_if_havent: {self.current_training_epoch, self.global_model_epoch, self.global_model_size, self.global_accuracy, self.global_loss,}")

    
        
class Server:
    def __init__(self, host=ARGS.host, port=ARGS.port):
        self.host = host
        self.port = port

        connection_database = self.read_or_initiate_database()

        self.clients = {}  # Maps UIDs to client sockets
        for uid in connection_database:
            self.clients[uid] = {
                'socket': None,
                'connection_status': 'DISCONNECTED'
            }

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))

        self.centralized_fl = CentralizeFL()

    def write_database(self, data):
        with open(CONNECTION_DATABASE_PATH, 'w') as file:
            json.dump(data, file, indent=2)

    def read_or_initiate_database(self):
        if not os.path.isfile(CONNECTION_DATABASE_PATH):
            data = []
            self.write_database(data)
        else:
            with open(CONNECTION_DATABASE_PATH, 'r') as file:
                data = json.load(file)
        
        return data
    
    def make_backup(self):
        self.write_database(list(self.clients.keys()))

    def handle_unrecognized_message(self, message):
        print(f"Unable to recognize message: {message}")

    def receive_data_from_specific_client(self, client_uid, data_size):
        received_data = b''
        while len(received_data) < data_size:
            try:
                more_data = self.clients[client_uid]['socket'].recv(data_size - len(received_data))
                if not more_data:
                    raise Exception("Server closed the connection unexpectedly.")
                received_data += more_data
            except socket.timeout:
                raise Exception("Timed out waiting for data from server.")
            
        return received_data

    def receive_client_training_result(self, client_uid, client_model_size):
        client_model_payload = self.receive_data_from_specific_client(client_uid, client_model_size)
        client_model = msgpack.unpackb(client_model_payload, object_hook=msgpack_numpy.decode)

        self.send_to_specific_client(client_uid, f"SERVER_CONFIRM_RECEIVED_MODEL:::{sha256_hash(client_model_payload)}")

        if client_model['epoch'] != self.centralized_fl.current_training_epoch:
            print("Mismatch epoch, reject model")
            return "MODEL_REJECTED_MISMATCH_EPOCH"
        if client_uid in self.centralized_fl.client_model_record.keys():
            print("Already exist, reject model")
            return "MODEL_REJECTED_ALREADY_EXIST"
        
        self.centralized_fl.receive_client_result(client_uid, client_model)

        return "MODEL_RECEIVED"

    def send_global_model_to_client(self, client_uid, message):
        if int(message.split(':::')[1]) != self.centralized_fl.global_model_size:
            raise Exception("Something wrong with confirming the global model size. Can't transfer model")

        print(f"Start transfering to {client_uid}")
        start_time = time.time()
        self.transfer_data_to_specific_client(client_uid, self.centralized_fl.global_model_params_payload)
        print(f"Complete global model transfer to {client_uid} at: {time.time() - start_time}")

        message = self.clients[client_uid]['socket'].recv(1024).decode('utf-8')

        if not message.startswith("CLIENT_CONFIRM_MODEL_RECEIVED") or message.split(":::")[1] != sha256_hash(self.centralized_fl.global_model_params_payload):
            raise Exception("The client-received model does not match")
        print(f"Complete global model confirm at: {time.time() - start_time}")

    def handle_client_send_status(self, client_uid, message):
        # print(f"Receive message from {client_uid}: {message}")
        client_status = json.loads(message.split(':::')[1])

        server_status = {
            'client_uid': client_uid,
        }

        if self.centralized_fl.aggregation_running:
            self.centralized_fl.get_aggregated_model_if_havent()

        if client_status['status'] == 'TRAINING_IN_PROGRESS':
            server_status['command'] = 'KEEP_GOING' # IT'S NOT THAT IMPORTANT!!!!!!!!!
        elif client_status['status'] == 'TRAINING_COMPLETED':
            if client_status['epoch'] == self.centralized_fl.current_training_epoch:
                if client_uid in self.centralized_fl.client_model_record or self.centralized_fl.aggregation_running:
                    server_status['command'] = 'STAY_PUT'
                else:
                    server_status['command'] = 'SEND_LOCAL_MODEL'
                    server_status['local_model_payload_size'] = client_status['local_model_payload_size']
                    server_status['current_training_epoch'] = self.centralized_fl.current_training_epoch
                    print(json.dumps(server_status, indent=2))
            else:
                if self.centralized_fl.aggregation_running: # Indicate that there is a global model to be distributed
                    server_status['command'] = 'STAY_PUT'
                else:
                    server_status['command'] = 'START_NEW_LOCAL_TRAINING'
                    server_status['current_training_epoch'] = self.centralized_fl.current_training_epoch
                    server_status['global_model_size'] = self.centralized_fl.global_model_size
        elif client_status['status'] == 'NO_TRAINING_RESULT':
            if self.centralized_fl.aggregation_running:
                server_status['command'] = 'STAY_PUT'
            else:
                server_status['command'] = 'START_NEW_LOCAL_TRAINING'
                server_status['current_training_epoch'] = self.centralized_fl.current_training_epoch
                server_status['global_model_size'] = self.centralized_fl.global_model_size
        else:
            raise Exception("Something's wrong with client status")
        
        self.send_to_specific_client(client_uid, f"SERVER_SEND_STATUS:::{json.dumps(server_status)}")

        if server_status['command'] == 'SEND_LOCAL_MODEL':
            self.receive_client_training_result(client_uid, client_status['local_model_payload_size'])
        

    def create_new_client(self, client_socket):
        client_uid = str(uuid.uuid4())

        self.clients[client_uid] = {
            'socket': client_socket,
            'connection_status': 'CONNECTED'
        }
        self.make_backup()
        
        client_socket.send(f"SERVER_ASSIGN_NEW_CLIENT_ID:{client_uid}".encode('utf-8'))

        return client_uid

    def update_existing_client(self, client_socket, message):
        client_uid = message.split(':')[1]

        if client_uid in self.clients:
            self.clients[client_uid]['socket'] = client_socket
            self.clients[client_uid]['connection_status'] = 'CONNECTED'

            client_socket.send(f"SERVER_ACK_EXISTING_CLIENT:{client_uid}".encode('utf-8'))

            return client_uid
        else:
            raise Exception(f"Client uid not accepted: {message}")
        
    def check_client_uid_ack(self, client_uid, message):
        ack_uid = message.split(':')[1]
        if ack_uid != client_uid:
            raise Exception(f"Ack client uid failed: {message}")

    def hand_shake(self, client_socket, client_addr):
        # Stage 1: Server ask for identity
        client_socket.send(f"SERVER_IDENTIFY_CLIENT".encode('utf-8'))


        # Stage 2: Client response with existing id or new id request
        message = client_socket.recv(1024).decode('utf-8')
        print(message)
        
        if message == "NEW_CLIENT_REQUEST_ID":
            client_uid = self.create_new_client(client_socket)
        elif message.startswith("EXISTING_CLIENT_SUBMIT_ID"):
            client_uid = self.update_existing_client(client_socket, message)

        # Stage 3: Client response with acknowledging the id
        message = client_socket.recv(1024).decode('utf-8')
        print(message)
        if message.startswith("NEW_CLIENT_SUBMIT_ACK"):
            self.check_client_uid_ack(client_uid, message)
            print(f"New client joined the network. Connection established. UID: {client_uid}")
        elif message.startswith("EXISTING_CLIENT_SUBMIT_ACK"):
            self.check_client_uid_ack(client_uid, message)
            print(f"Existing client reconnected. UID: {client_uid}")
        else:
            raise Exception(f"Handshake failed at: {message}")
        
        return client_uid

    def handle_client_connection(self, client_socket, client_addr):
        # client_socket.settimeout(5.0)
        client_uid = None

        try:
            client_uid = self.hand_shake(client_socket, client_addr)
        except Exception as e:
            print(f"Error: {e}")

        if client_uid is not None:
            while True:
                try:
                    message = client_socket.recv(1024).decode('utf-8')
                    if message:
                        if message.startswith("CLIENT_SEND_STATUS"):
                            self.handle_client_send_status(client_uid, message)
                        elif message.startswith("CLIENT_INITIATE_GLOBAL_MODEL_RECEIVE"):
                            self.send_global_model_to_client(client_uid, message)
                        else:
                            self.handle_unrecognized_message(message)
                    else:
                        break
                except Exception as e:
                    print(f"Error: {e}")
                    break

        print(f"Connection closed. Address: {client_addr}. Client: {client_uid}")
        if client_uid is not None and client_uid in self.clients:
            del self.clients[client_uid]['socket']
            self.clients[client_uid]['connection_status'] = 'DISCONNECTED'
        client_socket.close()

    def send_to_specific_client(self, uid, message):
        # print(f"Sent to {uid}: {message}")
        self.transfer_data_to_specific_client(uid, message.encode('utf-8'))

    def transfer_data_to_specific_client(self, uid, data):
        if uid in self.clients and self.clients[uid]['connection_status'] == 'CONNECTED':
            self.clients[uid]['socket'].sendall(data)
        else:
            print(f"Client UID {uid} not found or DISCONNECTED.")

    def run(self):
        self.server_socket.listen()
        print(f"Server listening on {self.host}:{self.port}")
        print("Server running. Ctrl+C to stop.")

        try:
            while True:
                client_socket, client_addr = self.server_socket.accept()
                thread = threading.Thread(target=self.handle_client_connection, args=(client_socket, client_addr))
                thread.start()
        except KeyboardInterrupt:
            print("Server stopping...")
        finally:
            self.server_socket.close()

if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver')
    server = Server()
    server.run()

    # centralized_fl = CentralizeFL()
    # centralized_fl.start_aggregation_if_suffient_result()

    # server.handle_client_send_status('272b67e5-ac67-4860-9d97-69cdf043acdb', "CLIENT_SEND_STATUS:::{'epoch': 6, 'status': 'TRAINING_COMPLETED', 'local_model_payload_size': 515276111}")
    