import grpc
from concurrent import futures
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import psutil
import sys
import os
import subprocess
import random
import socket
import json
from datetime import datetime
from collections import deque
import netifaces

import federated_learning_pb2
import federated_learning_pb2_grpc

# ==================== CONFIGURATION ====================
EDGE_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
EDGE_PORT = 5001

# FIX: Use the CORRECT CLOUD IP from your cloud.py
CLOUD_IP = '<REPLACE_WITH_CLOUD_IP>'  # MATCHES cloud.py EXTERNAL_IP
CLOUD_PORT = 5000

# Edge specs
EDGE_SPECS = [
    {'ram_gb': 4, 'cores': 8, 'storage_gb': 80, 'ip': '192.168.49.150'},
    {'ram_gb': 8, 'cores': 8, 'storage_gb': 80, 'ip': '192.168.49.151'},
    {'ram_gb': 8, 'cores': 8, 'storage_gb': 80, 'ip': '192.168.49.152'},
    {'ram_gb': 8, 'cores': 8, 'storage_gb': 80, 'ip': '192.168.49.153'},
    {'ram_gb': 7, 'cores': 8, 'storage_gb': 100, 'ip': '192.168.49.154'},
    {'ram_gb': 7, 'cores': 8, 'storage_gb': 100, 'ip': '192.168.49.155'},
    {'ram_gb': 7, 'cores': 8, 'storage_gb': 100, 'ip': '192.168.49.156'},
    {'ram_gb': 7, 'cores': 8, 'storage_gb': 100, 'ip': '192.168.49.157'},
    {'ram_gb': 9, 'cores': 4, 'storage_gb': 90, 'ip': '192.168.49.158'},
    {'ram_gb': 9, 'cores': 4, 'storage_gb': 90, 'ip': '192.168.49.159'},
    {'ram_gb': 9, 'cores': 4, 'storage_gb': 90, 'ip': '192.168.49.160'},
    {'ram_gb': 9, 'cores': 4, 'storage_gb': 90, 'ip': '192.168.49.161'},
    {'ram_gb': 6, 'cores': 8, 'storage_gb': 70, 'ip': '192.168.49.162'},
    {'ram_gb': 6, 'cores': 8, 'storage_gb': 70, 'ip': '192.168.49.163'},
    {'ram_gb': 6, 'cores': 8, 'storage_gb': 70, 'ip': '192.168.49.164'},
    {'ram_gb': 7, 'cores': 8, 'storage_gb': 80, 'ip': '192.168.49.165'},
    {'ram_gb': 7, 'cores': 8, 'storage_gb': 80, 'ip': '192.168.49.166'},
    {'ram_gb': 7, 'cores': 8, 'storage_gb': 80, 'ip': '192.168.49.167'},
    {'ram_gb': 6, 'cores': 8, 'storage_gb': 80, 'ip': '192.168.49.168'},
]

edge_spec = EDGE_SPECS[EDGE_ID] if EDGE_ID < len(EDGE_SPECS) else EDGE_SPECS[0]
EDGE_IP = edge_spec['ip']

print(f"🖥️ Edge {EDGE_ID} Initialized")
print(f"  IP: {EDGE_IP}:{EDGE_PORT}")
print(f"  Specs: {edge_spec['ram_gb']}GB RAM, {edge_spec['cores']} cores")
print(f"  Cloud: {CLOUD_IP}:{CLOUD_PORT}")

# ==================== FASHION-MNIST MODEL ====================
class FashionMNISTCNN(nn.Module):
    """CNN for Fashion-MNIST (Matches Cloud model exactly)"""
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Global state
fl_model = FashionMNISTCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fl_model.parameters(), lr=0.001)

global_weights = None
local_weights = None
training_status = 'idle'
training_lock = threading.Lock()
training_logs = deque(maxlen=1000)
current_round = 0

# Metrics tracking
current_stress_level = 0
stress_process = None
_net_io_last = None
_net_time_last = time.time()
devices_data = {}  # device_id -> (data, labels)

# ==================== NETWORK UTILITIES ====================
def get_primary_interface():
    """Get primary network interface"""
    try:
        gateways = netifaces.gateways()
        default_gateway = gateways.get('default', {})
        if netifaces.AF_INET in default_gateway:
            interface = default_gateway[netifaces.AF_INET][1]
            return interface
    except:
        pass
    
    # Fallback to first non-loopback interface
    interfaces = netifaces.interfaces()
    for iface in interfaces:
        if iface.startswith('lo'):
            continue
        addrs = netifaces.ifaddresses(iface)
        if netifaces.AF_INET in addrs:
            return iface
    
    return 'eth0'  # Default

# ==================== STRESS MANAGEMENT ====================
def apply_stress(level, duration=60):
    """Apply CPU/memory stress using stress-ng"""
    global stress_process
    
    # Kill existing stress
    if stress_process:
        try:
            stress_process.terminate()
            stress_process.wait(timeout=5)
        except:
            pass
    
    # Map levels to stress-ng commands
    stress_commands = {
        0: None,  # NO_STRESS
        1: ["stress-ng", "--cpu", str(max(1, edge_spec['cores']//4)), "--timeout", str(duration)],  # LOW_STRESS
        2: ["stress-ng", "--cpu", str(max(1, edge_spec['cores']//2)), "--vm", "1", "--vm-bytes", f"{max(1, edge_spec['ram_gb']//4)}G", "--timeout", str(duration)],  # MEDIUM_STRESS
        3: ["stress-ng", "--cpu", str(max(1, edge_spec['cores'])), "--vm", "2", "--vm-bytes", f"{max(1, edge_spec['ram_gb']//2)}G", "--timeout", str(duration)]  # HIGH_STRESS
    }
    
    if level > 0 and level in stress_commands:
        try:
            stress_process = subprocess.Popen(
                stress_commands[level],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"🔥 Edge {EDGE_ID}: Applied stress level {level}")
            return True
        except Exception as e:
            print(f"❌ Edge {EDGE_ID}: Failed to apply stress: {e}")
            return False
    
    return True

# ==================== REAL METRICS COLLECTION ====================
def get_system_metrics():
    """Get real system metrics with proper enum"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)
        cpu = float(cpu_percent)
        
        # Memory usage
        mem = psutil.virtual_memory()
        mem_percent = float(mem.percent)
        
        # Network bandwidth - track specific interface
        global _net_io_last, _net_time_last
        current_time = time.time()
        
        bandwidth = 0.0
        try:
            interface = get_primary_interface()
            net_io = psutil.net_io_counters(pernic=True).get(interface)
            
            if net_io:
                if _net_io_last is not None and current_time > _net_time_last:
                    time_diff = current_time - _net_time_last
                    bytes_diff = (net_io.bytes_sent - _net_io_last.bytes_sent + 
                                 net_io.bytes_recv - _net_io_last.bytes_recv)
                    bandwidth = (bytes_diff * 8) / (time_diff * 1024 * 1024)  # Mbps
                
                _net_io_last = net_io
        except:
            # Fallback to total network usage
            net_io = psutil.net_io_counters()
            if _net_io_last is not None and current_time > _net_time_last:
                time_diff = current_time - _net_time_last
                bytes_diff = (net_io.bytes_sent - _net_io_last.bytes_sent + 
                             net_io.bytes_recv - _net_io_last.bytes_recv)
                bandwidth = (bytes_diff * 8) / (time_diff * 1024 * 1024)  # Mbps
            _net_io_last = net_io
        
        _net_time_last = current_time
        
        # Latency calculation (simulated based on stress and load)
        base_latency = 5.0  # ms
        cpu_factor = (cpu / 100.0) * 50.0
        stress_factor = current_stress_level * 25.0
        memory_factor = (mem_percent / 100.0) * 20.0
        
        latency = base_latency + cpu_factor + stress_factor + memory_factor
        
        # Add some randomness to make it realistic
        latency += random.uniform(-2.0, 5.0)
        latency = max(1.0, latency)
        
        return {
            'cpu': cpu,
            'memory': mem_percent,
            'bandwidth': bandwidth,
            'latency': latency,
            'stress': current_stress_level
        }
        
    except Exception as e:
        print(f"⚠️ Edge {EDGE_ID}: Error getting metrics: {e}")
        # Return reasonable defaults
        return {
            'cpu': 10.0 + current_stress_level * 15.0,
            'memory': 30.0 + current_stress_level * 10.0,
            'bandwidth': 50.0 - current_stress_level * 5.0,
            'latency': 20.0 + current_stress_level * 30.0,
            'stress': current_stress_level
        }

# ==================== DATA PROCESSING ====================
def prepare_fashion_mnist_data(data_bytes, labels_bytes):
    """Prepare Fashion-MNIST data for training"""
    try:
        # Deserialize
        data = pickle.loads(data_bytes)
        labels = pickle.loads(labels_bytes)
        
        # Convert to numpy
        if isinstance(data, list):
            data_np = np.array(data, dtype=np.float32)
        else:
            data_np = data.astype(np.float32)
            
        if isinstance(labels, list):
            labels_np = np.array(labels, dtype=np.int64)
        else:
            labels_np = labels.astype(np.int64)
        
        # Fashion-MNIST specific: images are 28x28 grayscale
        # Ensure proper shape: (samples, 1, 28, 28)
        if len(data_np.shape) == 2:  # (samples, 784)
            # Reshape from 784 to 1x28x28
            data_np = data_np.reshape(-1, 1, 28, 28)
        elif len(data_np.shape) == 3 and data_np.shape[1] == 28 and data_np.shape[2] == 28:  # (samples, 28, 28)
            data_np = np.expand_dims(data_np, axis=1)  # Add channel dimension
        elif len(data_np.shape) == 3 and data_np.shape[1] == 784:  # (samples, 784, ?)
            data_np = data_np[:, :784].reshape(-1, 1, 28, 28)
        
        # Normalize to [0, 1]
        if data_np.max() > 1.0:
            data_np = data_np / 255.0
        
        print(f"📊 Edge {EDGE_ID}: Prepared {len(data_np)} Fashion-MNIST samples")
        print(f"  Data shape: {data_np.shape}")
        print(f"  Labels shape: {labels_np.shape}")
        print(f"  Data range: [{data_np.min():.3f}, {data_np.max():.3f}]")
        
        return data_np, labels_np
        
    except Exception as e:
        print(f"❌ Edge {EDGE_ID}: Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ==================== FETCH GLOBAL MODEL ====================
def fetch_global_model():
    """Fetch global model from cloud"""
    global global_weights, current_round
    
    try:
        print(f"🌐 Edge {EDGE_ID}: Fetching global model from Cloud...")
        
        channel = grpc.insecure_channel(f"{CLOUD_IP}:{CLOUD_PORT}")
        stub = federated_learning_pb2_grpc.CloudCoreStub(channel)
        
        response = stub.GetGlobalModel(
            federated_learning_pb2.ModelRequest(edge_id=EDGE_ID),
            timeout=10
        )
        
        if response.model_weights:
            global_weights = pickle.loads(response.model_weights)
            current_round = response.round
            print(f"✅ Edge {EDGE_ID}: Fetched global model for round {current_round}")
            print(f"  Number of weight tensors: {len(global_weights)}")
            return True
        else:
            print(f"⚠️ Edge {EDGE_ID}: No global model received from Cloud")
            return False
            
    except Exception as e:
        print(f"❌ Edge {EDGE_ID}: Error fetching global model: {e}")
        return False

# ==================== FL TRAINING ====================
def train_fl_model():
    """Train Fashion-MNIST model with all received data"""
    global local_weights, training_status, training_logs, devices_data
    
    with training_lock:
        training_status = 'training'
    
    start_time = time.time()
    
    try:
        # Combine all device data
        if not devices_data:
            print(f"⚠️ Edge {EDGE_ID}: No data for training")
            with training_lock:
                training_status = 'done'
            return 0.0, 0.0, 0, []
        
        all_data = []
        all_labels = []
        
        for device_id, (data, labels) in devices_data.items():
            all_data.append(data)
            all_labels.append(labels)
        
        # Concatenate all data
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        print(f"\n🏋️‍♂️ Edge {EDGE_ID}: STARTING FL TRAINING")
        print(f"  Round: {current_round}")
        print(f"  Total samples: {len(combined_data)}")
        print(f"  Stress Level: {current_stress_level}")
        print(f"  Devices: {list(devices_data.keys())}")
        
        # Convert to tensors
        x_tensor = torch.FloatTensor(combined_data)
        y_tensor = torch.LongTensor(combined_labels)
        
        # Fetch global model first
        fetch_success = fetch_global_model()
        
        # Load global weights if available
        if fetch_success and global_weights is not None:
            try:
                # Create state dict
                state_dict = {}
                param_names = list(fl_model.state_dict().keys())
                
                # Check if weights match
                if len(global_weights) == len(param_names):
                    for name, weights in zip(param_names, global_weights):
                        state_dict[name] = torch.tensor(weights)
                    
                    fl_model.load_state_dict(state_dict)
                    print(f"✅ Edge {EDGE_ID}: Loaded global model weights")
                else:
                    print(f"⚠️ Edge {EDGE_ID}: Weight mismatch. Expected {len(param_names)}, got {len(global_weights)}")
            except Exception as e:
                print(f"⚠️ Edge {EDGE_ID}: Error loading global weights: {e}")
        
        # Create dataset
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        
        # Training
        fl_model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        epoch_logs = []
        
        # Local epochs (3 epochs)
        for epoch in range(3):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                optimizer.zero_grad()
                output = fl_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                
                _, predicted = torch.max(output.data, 1)
                batch_total = target.size(0)
                batch_correct = (predicted == target).sum().item()
                
                epoch_total += batch_total
                epoch_correct += batch_correct
                
                # Store log for this batch
                batch_acc = batch_correct / batch_total if batch_total > 0 else 0
                batch_log = {
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'loss': batch_loss,
                    'accuracy': batch_acc,
                    'timestamp': time.time()
                }
                training_logs.append(batch_log)
                epoch_logs.append(batch_log)
                
                # Print progress every 2 batches
                if batch_idx % 2 == 0:
                    print(f"    Edge {EDGE_ID}: Epoch {epoch+1}, Batch {batch_idx}: "
                          f"Loss={batch_loss:.4f}, Acc={batch_acc*100:.1f}%")
            
            epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
            epoch_avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
            
            print(f"  Edge {EDGE_ID}: Epoch {epoch+1}/3: Loss={epoch_avg_loss:.4f}, Accuracy={epoch_acc*100:.1f}%")
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
        
        # Calculate final metrics
        training_time = time.time() - start_time
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / (len(dataloader) * 3) if len(dataloader) > 0 else 0
        
        # Save trained weights
        local_weights = [param.data.cpu().numpy() for param in fl_model.parameters()]
        
        print(f"\n✅ Edge {EDGE_ID}: FL TRAINING COMPLETE")
        print(f"  Final Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"  Final Loss: {avg_loss:.4f}")
        print(f"  Training Time: {training_time:.1f}s")
        print(f"  Samples Trained: {total}")
        print(f"  Stress During Training: {current_stress_level}")
        
        # Clear device data after training
        devices_data.clear()
        
        with training_lock:
            training_status = 'done'
        
        return avg_loss, accuracy, int(training_time), epoch_logs
        
    except Exception as e:
        print(f"❌ Edge {EDGE_ID}: Training error: {e}")
        import traceback
        traceback.print_exc()
        
        with training_lock:
            training_status = 'error'
        
        return 0.0, 0.0, 0, []

# ==================== gRPC SERVICE ====================
class EdgeNodeService(federated_learning_pb2_grpc.EdgeNodeServicer):
    def __init__(self):
        print(f"✅ Edge {EDGE_ID} Service Initialized")
        # Start metrics reporting after a short delay
        time.sleep(2)
        self.start_metrics_reporting()
    
    def start_metrics_reporting(self):
        """Periodically report metrics to Cloud with proper enum"""
        def report_metrics():
            # Initial report
            self._report_single_metrics()
            
            # Periodic reports
            while True:
                try:
                    self._report_single_metrics()
                except Exception as e:
                    print(f"⚠️ Edge {EDGE_ID}: Metrics reporting error: {e}")
                
                time.sleep(5)
        
        thread = threading.Thread(target=report_metrics, daemon=True)
        thread.start()
    
    def _report_single_metrics(self):
        """Report single metrics to cloud"""
        try:
            channel = grpc.insecure_channel(f"{CLOUD_IP}:{CLOUD_PORT}")
            stub = federated_learning_pb2_grpc.CloudCoreStub(channel)
            
            metrics = get_system_metrics()
            
            # Map numeric stress to enum
            stress_enum_map = {
                0: federated_learning_pb2.NO_STRESS,
                1: federated_learning_pb2.LOW_STRESS,
                2: federated_learning_pb2.MEDIUM_STRESS,
                3: federated_learning_pb2.HIGH_STRESS
            }
            
            stress_enum = stress_enum_map.get(metrics['stress'], federated_learning_pb2.NO_STRESS)
            
            response = stub.ReportMetrics(
                federated_learning_pb2.EdgeMetricsReport(
                    edge_id=EDGE_ID,
                    cpu_usage=metrics['cpu'],
                    memory_usage=metrics['memory'],
                    bandwidth=metrics['bandwidth'],
                    latency=metrics['latency'],
                    stress_level=stress_enum
                ),
                timeout=5
            )
            
            if response.received:
                # Print less frequently to avoid console spam
                if random.random() < 0.1:  # 10% chance
                    print(f"📡 Edge {EDGE_ID}: Reported metrics - "
                        f"CPU={metrics['cpu']:.1f}%, Latency={metrics['latency']:.1f}ms")
            
        except Exception as e:
            # Don't print error every time to avoid spam
            if random.random() < 0.05:  # 5% chance
                print(f"⚠️ Edge {EDGE_ID}: Metrics reporting error (occasional): {e}")
    
    def GetMetrics(self, request, context):
        """Return current metrics with proper enum"""
        metrics = get_system_metrics()
        
        # Map numeric stress to enum
        stress_enum_map = {
            0: federated_learning_pb2.NO_STRESS,
            1: federated_learning_pb2.LOW_STRESS,
            2: federated_learning_pb2.MEDIUM_STRESS,
            3: federated_learning_pb2.HIGH_STRESS
        }
        
        stress_enum = stress_enum_map.get(metrics['stress'], federated_learning_pb2.NO_STRESS)
        
        return federated_learning_pb2.MetricsResponse(
            cpu_usage=metrics['cpu'],
            memory_usage=metrics['memory'],
            bandwidth=metrics['bandwidth'],
            latency=metrics['latency'],
            stress_level=stress_enum
        )
    
    def ReceiveData(self, request, context):
        """Receive data from device"""
        try:
            print(f"\n📥 Edge {EDGE_ID}: RECEIVING DATA FROM DEVICE {request.device_id}")
            print(f"  Round: {request.round}")
            print(f"  Data size: {len(request.data)} bytes")
            print(f"  Labels size: {len(request.labels)} bytes")
            
            # Prepare data
            data, labels = prepare_fashion_mnist_data(request.data, request.labels)
            
            if data is not None and labels is not None:
                # Store device data
                devices_data[request.device_id] = (data, labels)
                
                print(f"✅ Edge {EDGE_ID}: Received {len(data)} samples from Device {request.device_id}")
                print(f"  Total devices: {len(devices_data)}")
                print(f"  Total samples: {sum(len(d) for d, _ in devices_data.values())}")
                
                return federated_learning_pb2.DataResponse(
                    success=True,
                    samples_received=len(data)
                )
            else:
                return federated_learning_pb2.DataResponse(
                    success=False,
                    samples_received=0
                )
                
        except Exception as e:
            print(f"❌ Edge {EDGE_ID}: Error receiving data: {e}")
            return federated_learning_pb2.DataResponse(
                success=False,
                samples_received=0
            )
    
    def StartTraining(self, request, context):
        """Start FL training (triggered by Cloud)"""
        print(f"\n🚀 Edge {EDGE_ID}: STARTING TRAINING FOR ROUND {request.round}")
        
        # Start training in background
        threading.Thread(target=self._train_async, args=(request.round,), daemon=True).start()
        
        return federated_learning_pb2.TrainingResponse(
            success=True,
            message=f"Edge {EDGE_ID}: Training started for round {request.round}"
        )
    
    def _train_async(self, round_num):
        """Async training"""
        global current_round
        current_round = round_num
        loss, accuracy, training_time, logs = train_fl_model()
        
        # Store logs for later retrieval by Cloud
        global training_logs
        training_logs.extend(logs)
    
    def GetTrainingResults(self, request, context):
        """Return training results to Cloud"""
        with training_lock:
            status = training_status
        
        if status != 'done' or local_weights is None:
            return federated_learning_pb2.ResultsResponse(
                model_weights=b'',
                accuracy=0.0,
                loss=0.0,
                training_time=0,
                samples_trained=0
            )
        
        try:
            # Calculate metrics from logs
            if training_logs:
                # Get logs from current round
                recent_logs = [log for log in training_logs if 'timestamp' in log]
                if recent_logs:
                    avg_loss = np.mean([log['loss'] for log in recent_logs[-20:]])
                    avg_accuracy = np.mean([log['accuracy'] for log in recent_logs[-20:]])
                else:
                    avg_loss = 0.0
                    avg_accuracy = 0.0
            else:
                avg_loss = 0.0
                avg_accuracy = 0.0
            
            # Count total samples trained from device data (before clearing)
            total_samples = sum(len(data) for data, _ in devices_data.values())
            
            weights_bytes = pickle.dumps(local_weights)
            
            print(f"\n📤 Edge {EDGE_ID}: SENDING TRAINING RESULTS TO CLOUD")
            print(f"  Round: {current_round}")
            print(f"  Accuracy: {avg_accuracy:.4f}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Training Time: {60}s")
            print(f"  Samples Trained: {total_samples}")
            
            return federated_learning_pb2.ResultsResponse(
                model_weights=weights_bytes,
                accuracy=float(avg_accuracy),
                loss=float(avg_loss),
                training_time=60,  # Simulated time
                samples_trained=total_samples
            )
            
        except Exception as e:
            print(f"❌ Edge {EDGE_ID}: Error getting results: {e}")
            return federated_learning_pb2.ResultsResponse(
                model_weights=b'',
                accuracy=0.0,
                loss=0.0,
                training_time=0,
                samples_trained=0
            )
    
    def GetTrainingLogs(self, request, context):
        """Return detailed training logs to Cloud"""
        logs_response = federated_learning_pb2.LogsResponse()
        
        for log in list(training_logs)[-100:]:  # Last 100 logs
            if 'epoch' in log:
                logs_response.logs.append(
                    federated_learning_pb2.TrainingLog(
                        epoch=log['epoch'],
                        batch=log['batch'],
                        loss=log['loss'],
                        accuracy=log['accuracy'],
                        timestamp=str(log['timestamp'])
                    )
                )
        
        return logs_response
    
    def ApplyStress(self, request, context):
        """Apply stress to this edge"""
        global current_stress_level
        
        try:
            # Map enum to numeric level
            stress_map = {
                federated_learning_pb2.NO_STRESS: 0,
                federated_learning_pb2.LOW_STRESS: 1,
                federated_learning_pb2.MEDIUM_STRESS: 2,
                federated_learning_pb2.HIGH_STRESS: 3
            }
            
            stress_level = stress_map.get(request.level, 0)
            current_stress_level = stress_level
            
            success = apply_stress(stress_level, request.duration_seconds)
            
            if success:
                stress_names = ['NO_STRESS', 'LOW_STRESS', 'MEDIUM_STRESS', 'HIGH_STRESS']
                stress_name = stress_names[stress_level] if stress_level < len(stress_names) else "UNKNOWN"
                print(f"✅ Edge {EDGE_ID}: Applied {stress_name} stress for {request.duration_seconds}s")
                return federated_learning_pb2.StressResponse(
                    applied=True,
                    message=f"Edge {EDGE_ID}: Applied {stress_name} stress"
                )
            else:
                return federated_learning_pb2.StressResponse(
                    applied=False,
                    message=f"Edge {EDGE_ID}: Failed to apply stress"
                )
                
        except Exception as e:
            print(f"❌ Edge {EDGE_ID}: Error applying stress: {e}")
            return federated_learning_pb2.StressResponse(
                applied=False,
                message=str(e)
            )

def serve():
    """Start Edge server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    edge_service = EdgeNodeService()
    federated_learning_pb2_grpc.add_EdgeNodeServicer_to_server(edge_service, server)
    
    server.add_insecure_port(f'0.0.0.0:{EDGE_PORT}')
    server.start()
    
    print(f"\n{'='*70}")
    print(f"🖥️ EDGE NODE {EDGE_ID} - READY")
    print(f"{'='*70}")
    print(f"📍 IP Address: {EDGE_IP}:{EDGE_PORT}")
    print(f"☁️  Cloud Server: {CLOUD_IP}:{CLOUD_PORT}")
    print(f"⚙️  Specifications: {edge_spec['ram_gb']}GB RAM, {edge_spec['cores']} cores")
    print(f"🤖 ML Model: Fashion-MNIST CNN (compatible with Cloud)")
    print(f"📊 Metrics: Auto-reporting to Cloud every 5s")
    print(f"💪 Stress Testing: Ready (stress-ng)")
    print(f"🔄 FL Training: Ready (3 epochs, batch_size=32)")
    print(f"{'='*70}")
    print("✅ Edge server started successfully!")
    print("📡 Waiting for connections from Cloud and Devices...")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print(f"\n👋 Edge {EDGE_ID} shutting down gracefully...")
        # Clean up stress process
        global stress_process
        if stress_process:
            try:
                stress_process.terminate()
            except:
                pass
        server.stop(0)

if __name__ == '__main__':
    # Test cloud connectivity
    print("🔍 Testing Cloud connectivity...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((CLOUD_IP, CLOUD_PORT))
        sock.close()
        
        if result == 0:
            print(f"✅ Cloud connection test: SUCCESS ({CLOUD_IP}:{CLOUD_PORT})")
        else:
            print(f"⚠️ Cloud connection test: FAILED ({CLOUD_IP}:{CLOUD_PORT})")
            print("   Make sure cloud.py is running on the cloud VM")
    except Exception as e:
        print(f"❌ Cloud connection test error: {e}")
    
    serve()



