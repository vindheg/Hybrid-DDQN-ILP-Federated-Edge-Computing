import grpc
import numpy as np
import pickle
import time
import random
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


import federated_learning_pb2
import federated_learning_pb2_grpc


# ==================== CONFIGURATION ====================
DEVICE_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CLOUD_IP = '<REPLACE_WITH_CLOUD_IP>'  # Replace with actual cloud IP
CLOUD_PORT = 5000
ROUNDS = 50
TRAIN_SPLIT = 0.7  # 70% for training, 30% for testing
SUBSET_SIZE = 100  # Samples to send to edge per round


print(f"📱 Device {DEVICE_ID} - REAL Fashion-MNIST Federated Learning")


# ==================== REAL FASHION-MNIST DATASET ====================
class FashionMNISTDataLoader:
    def __init__(self, device_id):
        self.device_id = device_id
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.class_distribution = None
       
        print(f"📦 Device {device_id}: Loading REAL Fashion-MNIST dataset...")
        self._load_real_dataset()
   
    def _load_real_dataset(self):
        """Load real Fashion-MNIST dataset with proper train/test split"""
        try:
            os.makedirs('./data', exist_ok=True)
           
            # Define transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
           
            # Download Fashion-MNIST dataset
            print(f"Device {self.device_id}: Downloading Fashion-MNIST dataset...")
            trainset = torchvision.datasets.FashionMNIST(
                root='./data', train=True, download=True, transform=transform)
           
            testset = torchvision.datasets.FashionMNIST(
                root='./data', train=False, download=True, transform=transform)
           
            print(f"✅ Device {self.device_id}: Dataset loaded - "
                  f"{len(trainset)} train, {len(testset)} test samples")
           
            # Convert to numpy arrays
            train_loader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
            test_loader = DataLoader(testset, batch_size=len(testset), shuffle=False)
           
            # Get all data
            train_data, train_labels = next(iter(train_loader))
            test_data, test_labels = next(iter(test_loader))
           
            # Convert to numpy
            train_data_np = train_data.numpy()
            train_labels_np = train_labels.numpy()
            test_data_np = test_data.numpy()
            test_labels_np = test_labels.numpy()
           
            # Create non-IID distribution per device (different class focus)
            self._create_non_iid_split(train_data_np, train_labels_np,
                                      test_data_np, test_labels_np)
           
            # Analyze class distribution
            self._analyze_class_distribution()
           
        except Exception as e:
            print(f"❌ Device {self.device_id}: Error loading Fashion-MNIST: {e}")
            print("⚠️ Creating synthetic backup data...")
            self._create_backup_data()
   
    def _create_non_iid_split(self, train_data, train_labels, test_data, test_labels):
        """Create non-IID data split for federated learning"""
        # Each device specializes in 2-3 classes (non-IID setting)
        class_groups = [
            [0, 1, 2],    # T-shirt/top, Trouser, Pullover
            [3, 4, 5],    # Dress, Coat, Sandal
            [6, 7, 8],    # Shirt, Sneaker, Bag
            [8, 9, 0],    # Bag, Ankle boot, T-shirt/top
            [1, 2, 3],    # Trouser, Pullover, Dress
            [4, 5, 6],    # Coat, Sandal, Shirt
            [7, 8, 9],    # Sneaker, Bag, Ankle boot
            [0, 3, 6],    # T-shirt/top, Dress, Shirt
            [1, 4, 7],    # Trouser, Coat, Sneaker
            [2, 5, 8]     # Pullover, Sandal, Bag
        ]
       
        device_group = class_groups[self.device_id % len(class_groups)]
       
        print(f"📊 Device {self.device_id}: Specializing in classes: {device_group}")
        print(f"  Class names: {self._get_class_names(device_group)}")
       
        # Filter training data for these classes
        train_mask = np.isin(train_labels, device_group)
        test_mask = np.isin(test_labels, device_group)
       
        # Get data for selected classes
        selected_train_data = train_data[train_mask]
        selected_train_labels = train_labels[train_mask]
        selected_test_data = test_data[test_mask]
        selected_test_labels = test_labels[test_mask]
       
        # Split into train/test (70/30)
        train_size = int(len(selected_train_data) * TRAIN_SPLIT)
       
        # Shuffle indices
        indices = np.random.permutation(len(selected_train_data))
       
        # Training data (70%)
        train_indices = indices[:train_size]
        self.train_data = selected_train_data[train_indices]
        self.train_labels = selected_train_labels[train_indices]
       
        # Testing data (30% of training + all test data)
        test_train_indices = indices[train_size:]
        test_from_train = selected_train_data[test_train_indices]
        test_labels_from_train = selected_train_labels[test_train_indices]
       
        # Combine test data
        self.test_data = np.concatenate([test_from_train, selected_test_data], axis=0)
        self.test_labels = np.concatenate([test_labels_from_train, selected_test_labels], axis=0)
       
        # Limit to reasonable sizes for edge offloading
        max_train_samples = 1000
        max_test_samples = 300
       
        if len(self.train_data) > max_train_samples:
            indices = np.random.choice(len(self.train_data), max_train_samples, replace=False)
            self.train_data = self.train_data[indices]
            self.train_labels = self.train_labels[indices]
       
        if len(self.test_data) > max_test_samples:
            indices = np.random.choice(len(self.test_data), max_test_samples, replace=False)
            self.test_data = self.test_data[indices]
            self.test_labels = self.test_labels[indices]
       
        # Convert to NHWC format (compatible with edge processing)
        if len(self.train_data.shape) == 4 and self.train_data.shape[1] == 1:
            self.train_data = np.transpose(self.train_data, (0, 2, 3, 1))
            self.test_data = np.transpose(self.test_data, (0, 2, 3, 1))
   
    def _get_class_names(self, class_indices):
        """Get Fashion-MNIST class names"""
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        return [class_names[i] for i in class_indices]
   
    def _analyze_class_distribution(self):
        """Analyze and print class distribution"""
        unique_train, counts_train = np.unique(self.train_labels, return_counts=True)
        unique_test, counts_test = np.unique(self.test_labels, return_counts=True)
       
        self.class_distribution = {
            'train': dict(zip(unique_train, counts_train)),
            'test': dict(zip(unique_test, counts_test))
        }
       
        print(f"📊 Device {self.device_id}: Data Statistics")
        print(f"  Training samples: {len(self.train_data)}")
        print(f"  Testing samples: {len(self.test_data)}")
       
        # Print class distribution
        class_names = self._get_class_names(range(10))
        print("  Training class distribution:")
        for class_idx, count in sorted(self.class_distribution['train'].items()):
            class_name = class_names[class_idx]
            percentage = (count / len(self.train_labels)) * 100
            print(f"    {class_name:15s}: {count:4d} samples ({percentage:5.1f}%)")
   
    def _create_backup_data(self):
        """Create synthetic data if real data loading fails"""
        print(f"⚡ Device {self.device_id}: Creating synthetic Fashion-MNIST-like data...")
       
        # Create data that resembles Fashion-MNIST (28x28 grayscale images)
        np.random.seed(self.device_id)
       
        # Training data (1000 samples)
        self.train_data = np.random.randn(1000, 28, 28, 1).astype(np.float32) * 0.5 + 0.5
       
        # Non-IID labels: focus on specific classes
        base_classes = [self.device_id % 10, (self.device_id + 1) % 10, (self.device_id + 2) % 10]
        self.train_labels = np.random.choice(base_classes, 1000)
       
        # Test data (300 samples)
        self.test_data = np.random.randn(300, 28, 28, 1).astype(np.float32) * 0.5 + 0.5
        self.test_labels = np.random.randint(0, 10, 300)
       
        print(f"📊 Device {self.device_id}: Created synthetic data - "
              f"{len(self.train_data)} train, {len(self.test_data)} test samples")
   
    def get_training_subset(self, subset_size=100):
        """Get subset of training data for offloading"""
        if self.train_data is None or len(self.train_data) == 0:
            # Create dummy data if none available
            return np.random.randn(subset_size, 28, 28, 1).astype(np.float32), \
                   np.random.randint(0, 10, subset_size).astype(np.int64)
       
        if subset_size > len(self.train_data):
            subset_size = len(self.train_data)
       
        # Select random subset
        indices = np.random.choice(len(self.train_data), subset_size, replace=False)
        return self.train_data[indices], self.train_labels[indices]
   
    def evaluate_model(self, model_weights):
        """Evaluate model on local test data"""
        if self.test_data is None or len(self.test_data) == 0:
            return 0.0, 0.0
       
        try:
            # Create model instance (same as edge model)
            model = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
           
            # Load weights
            state_dict = {}
            param_names = list(model.state_dict().keys())
           
            for i, (name, param) in enumerate(zip(param_names, model_weights)):
                if i < len(model_weights):
                    state_dict[name] = torch.from_numpy(param) if isinstance(param, np.ndarray) else param
           
            model.load_state_dict(state_dict)
            model.eval()
           
            # Convert test data to tensor (NCHW format)
            test_data_tensor = torch.FloatTensor(np.transpose(self.test_data, (0, 3, 1, 2)))
            test_labels_tensor = torch.LongTensor(self.test_labels)
           
            # Evaluate
            with torch.no_grad():
                outputs = model(test_data_tensor)
                loss = nn.CrossEntropyLoss()(outputs, test_labels_tensor).item()
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == test_labels_tensor).sum().item() / len(test_labels_tensor)
           
            return accuracy, loss
           
        except Exception as e:
            print(f"⚠️ Device {self.device_id}: Model evaluation error: {e}")
            return 0.0, 0.0


# ==================== DEVICE FL CLIENT ====================
class DeviceFLClient:
    def __init__(self, device_id):
        self.device_id = device_id
        print(f"📱 Initializing Device {device_id}...")
       
        # Load REAL Fashion-MNIST data
        self.dataset = FashionMNISTDataLoader(device_id)
       
        self.global_model_weights = None
        self.current_round = 0
        self.cloud_stub = None
        self.connected = False
        self.local_accuracy_history = []
        self.local_loss_history = []
       
        print(f"✅ Device {device_id} ready with REAL Fashion-MNIST data")
        print(f"  Training samples: {len(self.dataset.train_data)}")
        print(f"  Testing samples: {len(self.dataset.test_data)}")
        print(f"  Cloud server: {CLOUD_IP}:{CLOUD_PORT}")
   
    def connect_to_cloud(self):
        """Connect to cloud and register"""
        try:
            channel = grpc.insecure_channel(
                f"{CLOUD_IP}:{CLOUD_PORT}",
                options=[
                    ('grpc.keepalive_time_ms', 5000),
                    ('grpc.keepalive_timeout_ms', 3000),
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024)
                ]
            )
            self.cloud_stub = federated_learning_pb2_grpc.CloudCoreStub(channel)
           
            # Register with cloud
            response = self.cloud_stub.RegisterDevice(
                federated_learning_pb2.DeviceInfo(
                    device_id=self.device_id,
                    ip_address=f"192.168.61.{100 + self.device_id}",
                    port=5100 + self.device_id,
                    data_samples=len(self.dataset.train_data)
                ),
                timeout=10
            )
           
            if response.success:
                print(f"✅ Device {self.device_id}: Connected and registered with Cloud")
                self.connected = True
                return True
            else:
                print(f"❌ Device {self.device_id}: Registration failed: {response.message}")
                return False
               
        except Exception as e:
            print(f"❌ Device {self.device_id}: Cannot connect to Cloud: {e}")
            return False
   
    def wait_for_round(self, round_num):
        """Wait for cloud to be ready for this round"""
        print(f"⏳ Device {self.device_id}: Waiting for Cloud to open Round {round_num}...")
       
        max_wait_time = 300
        start_time = time.time()
       
        while time.time() - start_time < max_wait_time:
            try:
                response = self.cloud_stub.ReadyForRound(
                    federated_learning_pb2.ReadyRequest(
                        device_id=self.device_id,
                        round=round_num
                    ),
                    timeout=10
                )
               
                if response.acknowledged:
                    print(f"✅ Device {self.device_id}: Cloud accepted readiness for Round {round_num}")
                   
                    if response.wait:
                        print(f"⏳ Device {self.device_id}: Waiting for other devices...")
                        # Wait and retry periodically
                        for i in range(60):
                            time.sleep(2)
                            try:
                                check_response = self.cloud_stub.ReadyForRound(
                                    federated_learning_pb2.ReadyRequest(
                                        device_id=self.device_id,
                                        round=round_num
                                    ),
                                    timeout=5
                                )
                                if not check_response.wait:
                                    print(f"✅ Device {self.device_id}: All devices ready!")
                                    return True
                            except:
                                pass
                        print(f"⚠️ Device {self.device_id}: Wait timeout, proceeding")
                    return True
                else:
                    print(f"⏳ Device {self.device_id}: Cloud not ready yet, waiting...")
                    time.sleep(10)
           
            except grpc.RpcError as e:
                print(f"⚠️ Device {self.device_id}: ReadyForRound error: {e.code()}, retrying...")
                time.sleep(5)
       
        print(f"❌ Device {self.device_id}: Timeout waiting for Round {round_num}")
        return False
   
    def evaluate_local_model(self):
        """Evaluate current global model on local test data"""
        if self.global_model_weights is None:
            return 0.0, 0.0
       
        accuracy, loss = self.dataset.evaluate_model(self.global_model_weights)
       
        self.local_accuracy_history.append(accuracy)
        self.local_loss_history.append(loss)
       
        print(f"📊 Device {self.device_id}: Local Evaluation - "
              f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
       
        return accuracy, loss
   
    def run_fl_round(self, round_num):
        print(f"\n{'='*70}")
        print(f"📅 DEVICE {self.device_id} - ROUND {round_num}/{ROUNDS}")
        print(f"{'='*70}")
       
        self.current_round = round_num
       
        try:
            # Ensure connection
            if not self.connected:
                if not self.connect_to_cloud():
                    return False
           
            # Step 1: Wait for round
            print(f"1. ⏳ Synchronizing with Cloud for Round {round_num}...")
            if not self.wait_for_round(round_num):
                return False
           
            # Step 2: Get edge assignment
            print("2. 🎯 Getting edge assignment...")
            try:
                assignment = self.cloud_stub.GetAssignment(
                    federated_learning_pb2.AssignmentRequest(
                        device_id=self.device_id,
                        round=round_num
                    ),
                    timeout=10
                )
               
                print(f"  ✅ Assigned to Edge {assignment.edge_id}")
                print(f"  📍 Edge address: {assignment.edge_ip}:{assignment.edge_port}")
               
            except grpc.RpcError as e:
                print(f"❌ Device {self.device_id}: GetAssignment error: {e.details()}")
                return False
           
            # Step 3: Send REAL Fashion-MNIST data to edge
            print("3. 📤 Sending REAL Fashion-MNIST data to Edge...")
           
            # Get subset of training data
            train_subset, labels_subset = self.dataset.get_training_subset(SUBSET_SIZE)
           
            print(f"  Sending {len(train_subset)} REAL Fashion-MNIST samples")
            print(f"  Data shape: {train_subset.shape}")
            print(f"  Labels: {np.unique(labels_subset, return_counts=True)}")
           
            try:
                edge_channel = grpc.insecure_channel(
                    f"{assignment.edge_ip}:{assignment.edge_port}",
                    options=[
                        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                        ('grpc.keepalive_time_ms', 5000)
                    ]
                )
                edge_stub = federated_learning_pb2_grpc.EdgeNodeStub(edge_channel)
               
                data_response = edge_stub.ReceiveData(
                    federated_learning_pb2.DataPayload(
                        device_id=self.device_id,
                        data=pickle.dumps(train_subset),
                        labels=pickle.dumps(labels_subset),
                        round=round_num
                    ),
                    timeout=15
                )
               
                if data_response.success:
                    print(f"  ✅ REAL Fashion-MNIST data sent: {data_response.samples_received} samples")
                else:
                    print(f"❌ Device {self.device_id}: Data transfer failed")
                    return False
                   
                # Trigger training on edge
                training_response = edge_stub.StartTraining(
                    federated_learning_pb2.TrainingRequest(round=round_num),
                    timeout=5
                )
               
                if training_response.success:
                    print(f"  ✅ Training triggered on Edge {assignment.edge_id}")
                else:
                    print(f"⚠️ Device {self.device_id}: Could not trigger training")
                   
            except Exception as e:
                print(f"❌ Device {self.device_id}: Data transfer error: {e}")
                import traceback
                traceback.print_exc()
                return False
           
            # Step 4: Wait for FL training
            print("4. ⏳ Waiting for FL training (60 seconds)...")
            time.sleep(60)
           
            # Step 5: Get updated model
            print("5. 🔄 Getting updated global model...")
            try:
                model_response = self.cloud_stub.GetGlobalModel(
                    federated_learning_pb2.ModelRequest(device_id=self.device_id),
                    timeout=10
                )
               
                if model_response.model_weights:
                    self.global_model_weights = pickle.loads(model_response.model_weights)
                    print(f"  ✅ Received global model for round {model_response.round}")
                   
                    # Evaluate model locally
                    local_accuracy, local_loss = self.evaluate_local_model()
                   
                    # Save model locally
                    self._save_local_model(round_num, local_accuracy, local_loss)
                else:
                    print(f"  ⚠️ Device {self.device_id}: No model received")
            except Exception as e:
                print(f"❌ Device {self.device_id}: GetGlobalModel error: {e}")
                return False
           
            print(f"✅ Device {self.device_id}: Round {round_num} completed successfully")
            return True
           
        except Exception as e:
            print(f"❌ Device {self.device_id}: Unexpected error in round {round_num}: {e}")
            import traceback
            traceback.print_exc()
            return False
   
    def _save_local_model(self, round_num, accuracy, loss):
        """Save local model with evaluation metrics"""
        try:
            os.makedirs(f'device_{self.device_id}_models', exist_ok=True)
            os.makedirs(f'device_{self.device_id}_results', exist_ok=True)
           
            # Save model
            model_data = {
                'weights': self.global_model_weights,
                'round': round_num,
                'accuracy': accuracy,
                'loss': loss,
                'timestamp': datetime.now().isoformat(),
                'device_id': self.device_id,
                'dataset_info': {
                    'train_samples': len(self.dataset.train_data),
                    'test_samples': len(self.dataset.test_data),
                    'class_distribution': self.dataset.class_distribution
                }
            }
           
            model_filename = f"device_{self.device_id}_models/model_round_{round_num:03d}.pth"
            torch.save(model_data, model_filename)
            print(f"💾 Device {self.device_id}: Model saved: {model_filename}")
           
            # Save results
            results = {
                'round': round_num,
                'accuracy': accuracy,
                'loss': loss,
                'accuracy_history': self.local_accuracy_history,
                'loss_history': self.local_loss_history,
                'timestamp': datetime.now().isoformat()
            }
           
            results_filename = f"device_{self.device_id}_results/results_round_{round_num:03d}.json"
            import json
            with open(results_filename, 'w') as f:
                json.dump(results, f, indent=2)
           
            # Plot progress (optional)
            self._plot_progress()
           
        except Exception as e:
            print(f"⚠️ Device {self.device_id}: Could not save model/results: {e}")
   
    def _plot_progress(self):
        """Plot learning progress (optional)"""
        try:
            if len(self.local_accuracy_history) > 1:
                import matplotlib.pyplot as plt
               
                plt.figure(figsize=(12, 4))
               
                # Accuracy plot
                plt.subplot(1, 2, 1)
                plt.plot(self.local_accuracy_history, 'b-o', linewidth=2, markersize=5)
                plt.xlabel('Round')
                plt.ylabel('Accuracy')
                plt.title(f'Device {self.device_id} - Local Accuracy')
                plt.grid(True, alpha=0.3)
               
                # Loss plot
                plt.subplot(1, 2, 2)
                plt.plot(self.local_loss_history, 'r-s', linewidth=2, markersize=5)
                plt.xlabel('Round')
                plt.ylabel('Loss')
                plt.title(f'Device {self.device_id} - Local Loss')
                plt.grid(True, alpha=0.3)
               
                plt.tight_layout()
                plt.savefig(f'device_{self.device_id}_results/progress.png', dpi=100, bbox_inches='tight')
                plt.close()
               
                print(f"📈 Device {self.device_id}: Progress plot saved")
        except:
            pass  # Optional feature
   
    def run_experiment(self):
        print(f"\n{'='*70}")
        print(f"🚀 DEVICE {self.device_id} - FL EXPERIMENT WITH REAL FASHION-MNIST")
        print(f"{'='*70}")
        print(f"📡 Cloud: {CLOUD_IP}:{CLOUD_PORT}")
        print(f"🎯 Rounds: {ROUNDS}")
        print(f"📊 Dataset: Fashion-MNIST (70/30 split)")
        print(f"📦 Samples per round: {SUBSET_SIZE}")
        print(f"{'='*70}")
       
        # Print dataset info
        print(f"\n📋 Device {self.device_id} Dataset Summary:")
        print(f"  Training samples: {len(self.dataset.train_data)}")
        print(f"  Testing samples: {len(self.dataset.test_data)}")
       
        if self.dataset.class_distribution:
            class_names = self.dataset._get_class_names(range(10))
            print(f"  Training class focus:")
            for class_idx, count in sorted(self.dataset.class_distribution['train'].items()):
                class_name = class_names[class_idx]
                percentage = (count / len(self.dataset.train_data)) * 100
                print(f"    {class_name:15s}: {count:4d} samples ({percentage:5.1f}%)")
       
        # Connect first
        print(f"\n🔗 Device {self.device_id}: Connecting to Cloud Core...")
        if not self.connect_to_cloud():
            print("❌ Failed to connect to Cloud. Exiting.")
            return
       
        successful_rounds = 0
       
        for round_num in range(1, ROUNDS + 1):
            success = self.run_fl_round(round_num)
           
            if success:
                successful_rounds += 1
                print(f"✅ Device {self.device_id}: Round {round_num} SUCCESS")
            else:
                print(f"❌ Device {self.device_id}: Round {round_num} FAILED")
           
            if round_num < ROUNDS:
                sleep_time = 15
                print(f"\n⏳ Device {self.device_id}: Waiting {sleep_time} seconds before next round...")
                time.sleep(sleep_time)
       
        # Final summary
        print(f"\n{'='*70}")
        print(f"📊 DEVICE {self.device_id} - EXPERIMENT COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Successful rounds: {successful_rounds}/{ROUNDS}")
       
        if self.local_accuracy_history:
            final_accuracy = self.local_accuracy_history[-1]
            best_accuracy = max(self.local_accuracy_history)
            avg_accuracy = np.mean(self.local_accuracy_history)
           
            print(f"\n📈 Learning Performance:")
            print(f"  Final accuracy: {final_accuracy:.4f}")
            print(f"  Best accuracy: {best_accuracy:.4f}")
            print(f"  Average accuracy: {avg_accuracy:.4f}")
           
            if successful_rounds > 1:
                improvement = ((final_accuracy - self.local_accuracy_history[0]) /
                             self.local_accuracy_history[0]) * 100
                print(f"  Improvement: {improvement:+.1f}%")
       
        print(f"\n💾 All models saved in: device_{self.device_id}_models/")
        print(f"📊 Results saved in: device_{self.device_id}_results/")
        print("🎉 Device participation completed!")


# ==================== MAIN ====================
def main():
    device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
   
    print(f"🚀 Starting Device {device_id} with REAL Fashion-MNIST...")
   
    # Check if gRPC modules exist
    try:
        import federated_learning_pb2
        import federated_learning_pb2_grpc
    except ImportError:
        print("❌ gRPC modules not found!")
        print("Run this command on ALL machines:")
        print("python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. federated_learning.proto")
        return
   
    # Check if PyTorch and torchvision are installed
    try:
        import torch
        import torchvision
    except ImportError:
        print("❌ PyTorch/torchvision not found!")
        print("Install with: pip install torch torchvision")
        return#!/usr/bin/env python3
# device.py - FIXED VERSION FOR CLOUD/EDGE COMPATIBILITY
import grpc
import numpy as np
import pickle
import time
import random
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import warnings
import socket
warnings.filterwarnings('ignore')


import federated_learning_pb2
import federated_learning_pb2_grpc


# ==================== CONFIGURATION ====================
DEVICE_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CLOUD_IP = '192.168.61.63'
CLOUD_PORT = 5000
ROUNDS = 50
TRAIN_SPLIT = 0.7  # 70% for training, 30% for testing
SUBSET_SIZE = 100  # Samples to send to edge per round


print(f"📱 Device {DEVICE_ID} - REAL Fashion-MNIST Federated Learning")


# ==================== COMPATIBLE MODEL ====================
class FashionMNISTCNN(nn.Module):
    """EXACTLY the same as cloud and edge models"""
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


# ==================== REAL FASHION-MNIST DATASET ====================
class FashionMNISTDataLoader:
    def __init__(self, device_id):
        self.device_id = device_id
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.class_distribution = None
       
        print(f"📦 Device {device_id}: Loading REAL Fashion-MNIST dataset...")
        self._load_real_dataset()
   
    def _load_real_dataset(self):
        """Load real Fashion-MNIST dataset with proper train/test split"""
        try:
            os.makedirs('./data', exist_ok=True)
           
            # Define transforms - keep as tensor for NCHW format
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
           
            # Download Fashion-MNIST dataset
            print(f"Device {self.device_id}: Downloading Fashion-MNIST dataset...")
            trainset = torchvision.datasets.FashionMNIST(
                root='./data', train=True, download=True, transform=transform)
           
            testset = torchvision.datasets.FashionMNIST(
                root='./data', train=False, download=True, transform=transform)
           
            print(f"✅ Device {self.device_id}: Dataset loaded - "
                  f"{len(trainset)} train, {len(testset)} test samples")
           
            # Convert to numpy arrays (keeping NCHW format)
            train_loader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
            test_loader = DataLoader(testset, batch_size=len(testset), shuffle=False)
           
            # Get all data
            train_data, train_labels = next(iter(train_loader))
            test_data, test_labels = next(iter(test_loader))
           
            # Convert to numpy - KEEP as NCHW (1, 28, 28)
            self.train_data = train_data.numpy()
            self.train_labels = train_labels.numpy()
            self.test_data = test_data.numpy()
            self.test_labels = test_labels.numpy()
           
            # Create non-IID distribution per device (different class focus)
            self._create_non_iid_split()
           
            # Analyze class distribution
            self._analyze_class_distribution()
           
        except Exception as e:
            print(f"❌ Device {self.device_id}: Error loading Fashion-MNIST: {e}")
            print("⚠️ Creating synthetic backup data...")
            self._create_backup_data()
   
    def _create_non_iid_split(self):
        """Create non-IID data split for federated learning"""
        # Each device specializes in 2-3 classes (non-IID setting)
        class_groups = [
            [0, 1, 2],    # T-shirt/top, Trouser, Pullover
            [3, 4, 5],    # Dress, Coat, Sandal
            [6, 7, 8],    # Shirt, Sneaker, Bag
            [8, 9, 0],    # Bag, Ankle boot, T-shirt/top
            [1, 2, 3],    # Trouser, Pullover, Dress
            [4, 5, 6],    # Coat, Sandal, Shirt
            [7, 8, 9],    # Sneaker, Bag, Ankle boot
            [0, 3, 6],    # T-shirt/top, Dress, Shirt
            [1, 4, 7],    # Trouser, Coat, Sneaker
            [2, 5, 8]     # Pullover, Sandal, Bag
        ]
       
        device_group = class_groups[self.device_id % len(class_groups)]
       
        print(f"📊 Device {self.device_id}: Specializing in classes: {device_group}")
        print(f"  Class names: {self._get_class_names(device_group)}")
       
        # Filter training data for these classes
        train_mask = np.isin(self.train_labels, device_group)
        test_mask = np.isin(self.test_labels, device_group)
       
        # Get data for selected classes
        selected_train_data = self.train_data[train_mask]
        selected_train_labels = self.train_labels[train_mask]
        selected_test_data = self.test_data[test_mask]
        selected_test_labels = self.test_labels[test_mask]
       
        # Split into train/test (70/30)
        train_size = int(len(selected_train_data) * TRAIN_SPLIT)
       
        # Shuffle indices
        indices = np.random.permutation(len(selected_train_data))
       
        # Training data (70%)
        train_indices = indices[:train_size]
        self.train_data = selected_train_data[train_indices]
        self.train_labels = selected_train_labels[train_indices]
       
        # Testing data (30% of training + all test data)
        test_train_indices = indices[train_size:]
        test_from_train = selected_train_data[test_train_indices]
        test_labels_from_train = selected_train_labels[test_train_indices]
       
        # Combine test data
        self.test_data = np.concatenate([test_from_train, selected_test_data], axis=0)
        self.test_labels = np.concatenate([test_labels_from_train, selected_test_labels], axis=0)
       
        # Limit to reasonable sizes for edge offloading
        max_train_samples = 1000
        max_test_samples = 300
       
        if len(self.train_data) > max_train_samples:
            indices = np.random.choice(len(self.train_data), max_train_samples, replace=False)
            self.train_data = self.train_data[indices]
            self.train_labels = self.train_labels[indices]
       
        if len(self.test_data) > max_test_samples:
            indices = np.random.choice(len(self.test_data), max_test_samples, replace=False)
            self.test_data = self.test_data[indices]
            self.test_labels = self.test_labels[indices]
       
        print(f"📊 Device {self.device_id}: Final dataset sizes")
        print(f"  Train: {self.train_data.shape}")  # Should be (samples, 1, 28, 28)
        print(f"  Test: {self.test_data.shape}")
   
    def _get_class_names(self, class_indices):
        """Get Fashion-MNIST class names"""
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        return [class_names[i] for i in class_indices]
   
    def _analyze_class_distribution(self):
        """Analyze and print class distribution"""
        unique_train, counts_train = np.unique(self.train_labels, return_counts=True)
        unique_test, counts_test = np.unique(self.test_labels, return_counts=True)
       
        self.class_distribution = {
            'train': dict(zip(unique_train, counts_train)),
            'test': dict(zip(unique_test, counts_test))
        }
       
        print(f"📊 Device {self.device_id}: Data Statistics")
        print(f"  Training samples: {len(self.train_data)}")
        print(f"  Testing samples: {len(self.test_data)}")
       
        # Print class distribution
        class_names = self._get_class_names(range(10))
        print("  Training class distribution:")
        for class_idx, count in sorted(self.class_distribution['train'].items()):
            class_name = class_names[class_idx]
            percentage = (count / len(self.train_labels)) * 100
            print(f"    {class_name:15s}: {count:4d} samples ({percentage:5.1f}%)")
   
    def _create_backup_data(self):
        """Create synthetic data if real data loading fails"""
        print(f"⚡ Device {self.device_id}: Creating synthetic Fashion-MNIST-like data...")
       
        # Create data that resembles Fashion-MNIST (28x28 grayscale images)
        np.random.seed(self.device_id)
       
        # Training data (1000 samples) - NCHW format
        self.train_data = np.random.randn(1000, 1, 28, 28).astype(np.float32) * 0.5 + 0.5
       
        # Non-IID labels: focus on specific classes
        base_classes = [self.device_id % 10, (self.device_id + 1) % 10, (self.device_id + 2) % 10]
        self.train_labels = np.random.choice(base_classes, 1000)
       
        # Test data (300 samples)
        self.test_data = np.random.randn(300, 1, 28, 28).astype(np.float32) * 0.5 + 0.5
        self.test_labels = np.random.randint(0, 10, 300)
       
        print(f"📊 Device {self.device_id}: Created synthetic data - "
              f"{len(self.train_data)} train, {len(self.test_data)} test samples")
   
    def get_training_subset(self, subset_size=100):
        """Get subset of training data for offloading"""
        if self.train_data is None or len(self.train_data) == 0:
            # Create dummy data if none available
            dummy_data = np.random.randn(subset_size, 1, 28, 28).astype(np.float32)
            dummy_labels = np.random.randint(0, 10, subset_size).astype(np.int64)
            return dummy_data, dummy_labels
       
        if subset_size > len(self.train_data):
            subset_size = len(self.train_data)
       
        # Select random subset
        indices = np.random.choice(len(self.train_data), subset_size, replace=False)
       
        # Ensure data is in correct format for edge
        subset_data = self.train_data[indices]
        subset_labels = self.train_labels[indices]
       
        return subset_data, subset_labels
   
    def evaluate_model(self, model_weights):
        """Evaluate model on local test data"""
        if self.test_data is None or len(self.test_data) == 0:
            return 0.0, 0.0
       
        try:
            # Create EXACT same model as edge/cloud
            model = FashionMNISTCNN()
            model.eval()
           
            # Load weights
            if isinstance(model_weights, list):
                # Create state dict
                state_dict = {}
                param_names = list(model.state_dict().keys())
               
                for i, (name, param) in enumerate(zip(param_names, model_weights)):
                    if i < len(model_weights):
                        if isinstance(param, np.ndarray):
                            state_dict[name] = torch.from_numpy(param)
                        else:
                            state_dict[name] = param
               
                model.load_state_dict(state_dict, strict=False)
           
            # Convert test data to tensor (already NCHW)
            test_data_tensor = torch.FloatTensor(self.test_data)
            test_labels_tensor = torch.LongTensor(self.test_labels)
           
            # Evaluate
            with torch.no_grad():
                outputs = model(test_data_tensor)
                loss = nn.CrossEntropyLoss()(outputs, test_labels_tensor).item()
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == test_labels_tensor).sum().item() / len(test_labels_tensor)
           
            return accuracy, loss
           
        except Exception as e:
            print(f"⚠️ Device {self.device_id}: Model evaluation error: {e}")
            return 0.0, 0.0


# ==================== DEVICE FL CLIENT ====================
class DeviceFLClient:
    def __init__(self, device_id):
        self.device_id = device_id
        print(f"📱 Initializing Device {device_id}...")
       
        # Test Cloud connectivity first
        if not self._test_cloud_connectivity():
            print(f"❌ Device {device_id}: Cannot connect to Cloud at {CLOUD_IP}:{CLOUD_PORT}")
            print("   Make sure cloud.py is running on the Cloud VM")
            sys.exit(1)
       
        # Load REAL Fashion-MNIST data
        self.dataset = FashionMNISTDataLoader(device_id)
       
        self.global_model_weights = None
        self.current_round = 0
        self.cloud_stub = None
        self.connected = False
        self.local_accuracy_history = []
        self.local_loss_history = []
       
        print(f"✅ Device {device_id} ready with REAL Fashion-MNIST data")
        print(f"  Training samples: {len(self.dataset.train_data)}")
        print(f"  Testing samples: {len(self.dataset.test_data)}")
        print(f"  Cloud server: {CLOUD_IP}:{CLOUD_PORT}")
   
    def _test_cloud_connectivity(self):
        """Test if Cloud is reachable"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((CLOUD_IP, CLOUD_PORT))
            sock.close()
            return result == 0
        except:
            return False
   
    def connect_to_cloud(self):
        """Connect to cloud and register"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                channel = grpc.insecure_channel(
                    f"{CLOUD_IP}:{CLOUD_PORT}",
                    options=[
                        ('grpc.keepalive_time_ms', 5000),
                        ('grpc.keepalive_timeout_ms', 3000),
                        ('grpc.max_receive_message_length', 50 * 1024 * 1024)
                    ]
                )
                self.cloud_stub = federated_learning_pb2_grpc.CloudCoreStub(channel)
               
                # Test connection with timeout
                response = self.cloud_stub.RegisterDevice(
                    federated_learning_pb2.DeviceInfo(
                        device_id=self.device_id,
                        ip_address=f"192.168.61.{100 + self.device_id}",
                        port=5100 + self.device_id,
                        data_samples=len(self.dataset.train_data)
                    ),
                    timeout=10
                )
               
                if response.success:
                    print(f"✅ Device {self.device_id}: Connected and registered with Cloud")
                    self.connected = True
                    return True
                else:
                    print(f"❌ Device {self.device_id}: Registration failed: {response.message}")
                   
            except grpc.RpcError as e:
                print(f"⚠️ Device {self.device_id}: Connection attempt {attempt+1} failed: {e.code()}")
                if attempt < max_retries - 1:
                    time.sleep(2)
            except Exception as e:
                print(f"⚠️ Device {self.device_id}: Connection error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
       
        print(f"❌ Device {self.device_id}: Failed to connect after {max_retries} attempts")
        return False
   
    def wait_for_round(self, round_num):
        """Wait for cloud to be ready for this round - IMPROVED"""
        print(f"⏳ Device {self.device_id}: Synchronizing with Cloud for Round {round_num}...")
       
        max_attempts = 30  # 30 attempts = 60 seconds
        attempt = 0
       
        while attempt < max_attempts:
            attempt += 1
           
            try:
                response = self.cloud_stub.ReadyForRound(
                    federated_learning_pb2.ReadyRequest(
                        device_id=self.device_id,
                        round=round_num
                    ),
                    timeout=5
                )
               
                if response.acknowledged:
                    print(f"✅ Device {self.device_id}: Cloud accepted Round {round_num}")
                   
                    if response.wait:
                        print(f"⏳ Device {self.device_id}: Cloud says wait for other devices...")
                        # Wait a bit and check again
                        time.sleep(2)
                        continue
                    else:
                        print(f"✅ Device {self.device_id}: All devices ready!")
                        return True
                else:
                    print(f"⏳ Device {self.device_id}: Cloud not ready yet (attempt {attempt}/{max_attempts})")
                    time.sleep(2)
                   
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    print(f"⚠️ Device {self.device_id}: Cloud not reachable (attempt {attempt}/{max_attempts})")
                else:
                    print(f"⚠️ Device {self.device_id}: Cloud error {e.code()} (attempt {attempt}/{max_attempts})")
                time.sleep(2)
            except Exception as e:
                print(f"⚠️ Device {self.device_id}: Connection error (attempt {attempt}/{max_attempts}): {e}")
                time.sleep(2)
       
        print(f"❌ Device {self.device_id}: Failed to sync with Cloud after {max_attempts} attempts")
        return False
   
    def evaluate_local_model(self):
        """Evaluate current global model on local test data"""
        if self.global_model_weights is None:
            return 0.0, 0.0
       
        accuracy, loss = self.dataset.evaluate_model(self.global_model_weights)
       
        self.local_accuracy_history.append(accuracy)
        self.local_loss_history.append(loss)
       
        print(f"📊 Device {self.device_id}: Local Evaluation - "
              f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
       
        return accuracy, loss
   
    def send_data_to_edge(self, edge_ip, edge_port, data, labels, round_num, max_retries=3):
        """Send data to edge with retry logic"""
        for attempt in range(max_retries):
            try:
                edge_channel = grpc.insecure_channel(
                    f"{edge_ip}:{edge_port}",
                    options=[
                        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                        ('grpc.keepalive_time_ms', 5000)
                    ]
                )
                edge_stub = federated_learning_pb2_grpc.EdgeNodeStub(edge_channel)
               
                data_response = edge_stub.ReceiveData(
                    federated_learning_pb2.DataPayload(
                        device_id=self.device_id,
                        data=pickle.dumps(data),
                        labels=pickle.dumps(labels),
                        round=round_num
                    ),
                    timeout=15
                )
               
                if data_response.success:
                    print(f"✅ Device {self.device_id}: Data sent to edge: {data_response.samples_received} samples")
                    return True
                else:
                    print(f"❌ Device {self.device_id}: Edge data receive failed (attempt {attempt+1})")
                   
            except Exception as e:
                print(f"⚠️ Device {self.device_id}: Edge data transfer error (attempt {attempt+1}): {e}")
           
            if attempt < max_retries - 1:
                time.sleep(2)
       
        return False
   
    def run_fl_round(self, round_num):
        print(f"\n{'='*70}")
        print(f"📅 DEVICE {self.device_id} - ROUND {round_num}/{ROUNDS}")
        print(f"{'='*70}")
       
        self.current_round = round_num
       
        try:
            # Ensure connection
            if not self.connected:
                if not self.connect_to_cloud():
                    print(f"❌ Device {self.device_id}: Cannot connect to Cloud")
                    return False
           
            # Step 1: Wait for round
            print(f"1. ⏳ Synchronizing with Cloud for Round {round_num}...")
            if not self.wait_for_round(round_num):
                print(f"❌ Device {self.device_id}: Failed to sync with Cloud")
                return False
           
            # Step 2: Get edge assignment
            print("2. 🎯 Getting edge assignment...")
            try:
                assignment = self.cloud_stub.GetAssignment(
                    federated_learning_pb2.AssignmentRequest(
                        device_id=self.device_id,
                        round=round_num
                    ),
                    timeout=10
                )
               
                print(f"  ✅ Assigned to Edge {assignment.edge_id}")
                print(f"  📍 Edge address: {assignment.edge_ip}:{assignment.edge_port}")
               
            except grpc.RpcError as e:
                print(f"❌ Device {self.device_id}: GetAssignment error: {e.details()}")
                return False
           
            # Step 3: Send REAL Fashion-MNIST data to edge
            print("3. 📤 Sending REAL Fashion-MNIST data to Edge...")
           
            # Get subset of training data
            train_subset, labels_subset = self.dataset.get_training_subset(SUBSET_SIZE)
           
            print(f"  Sending {len(train_subset)} REAL Fashion-MNIST samples")
            print(f"  Data shape: {train_subset.shape}")  # Should be (N, 1, 28, 28)
            print(f"  Unique labels: {np.unique(labels_subset, return_counts=True)}")
           
            # Send data with retry
            if not self.send_data_to_edge(assignment.edge_ip, assignment.edge_port,
                                        train_subset, labels_subset, round_num):
                print(f"❌ Device {self.device_id}: Failed to send data to edge after retries")
                return False
           
            # Step 4: Trigger training on edge
            print("4. 🚀 Triggering training on edge...")
            try:
                edge_channel = grpc.insecure_channel(f"{assignment.edge_ip}:{assignment.edge_port}")
                edge_stub = federated_learning_pb2_grpc.EdgeNodeStub(edge_channel)
               
                training_response = edge_stub.StartTraining(
                    federated_learning_pb2.TrainingRequest(round=round_num),
                    timeout=5
                )
               
                if training_response.success:
                    print(f"  ✅ Training triggered on Edge {assignment.edge_id}")
                else:
                    print(f"⚠️ Device {self.device_id}: Edge training response: {training_response.message}")
                   
            except Exception as e:
                print(f"⚠️ Device {self.device_id}: Training trigger error: {e}")
                # Continue anyway, edge might auto-start training
           
            # Step 5: Wait for FL training
            #print("5. ⏳ Waiting for FL training (65 seconds)...")
            #time.sleep(65)  # Give extra 5 seconds for edge processing
           
            # Step 6: Get updated model
            print("6. 🔄 Getting updated global model...")
            try:
                model_response = self.cloud_stub.GetGlobalModel(
                    federated_learning_pb2.ModelRequest(device_id=self.device_id),
                    timeout=10
                )
               
                if model_response.model_weights:
                    self.global_model_weights = pickle.loads(model_response.model_weights)
                    print(f"  ✅ Received global model for round {model_response.round}")
                   
                    # Evaluate model locally
                    local_accuracy, local_loss = self.evaluate_local_model()
                   
                    # Save model locally
                    self._save_local_model(round_num, local_accuracy, local_loss)
                else:
                    print(f"  ⚠️ Device {self.device_id}: No model received")
            except Exception as e:
                print(f"❌ Device {self.device_id}: GetGlobalModel error: {e}")
                return False
           
            print(f"✅ Device {self.device_id}: Round {round_num} completed successfully")
            return True
           
        except Exception as e:
            print(f"❌ Device {self.device_id}: Unexpected error in round {round_num}: {e}")
            import traceback
            traceback.print_exc()
            return False
   
    def _save_local_model(self, round_num, accuracy, loss):
        """Save local model with evaluation metrics"""
        try:
            os.makedirs(f'device_{self.device_id}_models', exist_ok=True)
            os.makedirs(f'device_{self.device_id}_results', exist_ok=True)
           
            # Save model
            model_data = {
                'weights': self.global_model_weights,
                'round': round_num,
                'accuracy': accuracy,
                'loss': loss,
                'timestamp': datetime.now().isoformat(),
                'device_id': self.device_id,
                'dataset_info': {
                    'train_samples': len(self.dataset.train_data),
                    'test_samples': len(self.dataset.test_data),
                    'class_distribution': self.dataset.class_distribution
                }
            }
           
            model_filename = f"device_{self.device_id}_models/model_round_{round_num:03d}.pth"
            torch.save(model_data, model_filename)
            print(f"💾 Device {self.device_id}: Model saved: {model_filename}")
           
            # Save results
            results = {
                'round': round_num,
                'accuracy': accuracy,
                'loss': loss,
                'accuracy_history': self.local_accuracy_history,
                'loss_history': self.local_loss_history,
                'timestamp': datetime.now().isoformat()
            }
           
            results_filename = f"device_{self.device_id}_results/results_round_{round_num:03d}.json"
            import json
            with open(results_filename, 'w') as f:
                json.dump(results, f, indent=2)
           
            # Plot progress (optional)
            self._plot_progress()
           
        except Exception as e:
            print(f"⚠️ Device {self.device_id}: Could not save model/results: {e}")
   
    def _plot_progress(self):
        """Plot learning progress (optional)"""
        try:
            if len(self.local_accuracy_history) > 1:
                import matplotlib.pyplot as plt
               
                plt.figure(figsize=(12, 4))
               
                # Accuracy plot
                plt.subplot(1, 2, 1)
                plt.plot(self.local_accuracy_history, 'b-o', linewidth=2, markersize=5)
                plt.xlabel('Round')
                plt.ylabel('Accuracy')
                plt.title(f'Device {self.device_id} - Local Accuracy')
                plt.grid(True, alpha=0.3)
               
                # Loss plot
                plt.subplot(1, 2, 2)
                plt.plot(self.local_loss_history, 'r-s', linewidth=2, markersize=5)
                plt.xlabel('Round')
                plt.ylabel('Loss')
                plt.title(f'Device {self.device_id} - Local Loss')
                plt.grid(True, alpha=0.3)
               
                plt.tight_layout()
                plt.savefig(f'device_{self.device_id}_results/progress.png', dpi=100, bbox_inches='tight')
                plt.close()
               
                print(f"📈 Device {self.device_id}: Progress plot saved")
        except ImportError:
            pass  # matplotlib not installed
        except Exception as e:
            print(f"⚠️ Device {self.device_id}: Plot error: {e}")
   
    def run_experiment(self):
        print(f"\n{'='*70}")
        print(f"🚀 DEVICE {self.device_id} - FL EXPERIMENT WITH REAL FASHION-MNIST")
        print(f"{'='*70}")
        print(f"📡 Cloud: {CLOUD_IP}:{CLOUD_PORT}")
        print(f"🎯 Rounds: {ROUNDS}")
        print(f"📊 Dataset: Fashion-MNIST (70/30 split)")
        print(f"📦 Samples per round: {SUBSET_SIZE}")
        print(f"{'='*70}")
       
        # Print dataset info
        print(f"\n📋 Device {self.device_id} Dataset Summary:")
        print(f"  Training samples: {len(self.dataset.train_data)}")
        print(f"  Testing samples: {len(self.dataset.test_data)}")
       
        if self.dataset.class_distribution:
            class_names = self.dataset._get_class_names(range(10))
            print(f"  Training class focus:")
            for class_idx, count in sorted(self.dataset.class_distribution['train'].items()):
                class_name = class_names[class_idx]
                percentage = (count / len(self.dataset.train_data)) * 100
                print(f"    {class_name:15s}: {count:4d} samples ({percentage:5.1f}%)")
       
        # Connect first
        print(f"\n🔗 Device {self.device_id}: Connecting to Cloud Core...")
        if not self.connect_to_cloud():
            print("❌ Failed to connect to Cloud. Exiting.")
            return
       
        successful_rounds = 0
       
        for round_num in range(1, ROUNDS + 1):
            print(f"\n{'='*70}")
            print(f"🔄 Starting Round {round_num}/{ROUNDS}")
            print(f"{'='*70}")
           
            success = self.run_fl_round(round_num)
           
            if success:
                successful_rounds += 1
                print(f"✅ Device {self.device_id}: Round {round_num} SUCCESS")
            else:
                print(f"❌ Device {self.device_id}: Round {round_num} FAILED")
           
            if round_num < ROUNDS:
                sleep_time = 15
                print(f"\n⏳ Device {self.device_id}: Waiting {sleep_time} seconds before next round...")
                time.sleep(sleep_time)
       
        # Final summary
        print(f"\n{'='*70}")
        print(f"📊 DEVICE {self.device_id} - EXPERIMENT COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Successful rounds: {successful_rounds}/{ROUNDS}")
       
        if self.local_accuracy_history:
            final_accuracy = self.local_accuracy_history[-1] if self.local_accuracy_history else 0
            best_accuracy = max(self.local_accuracy_history) if self.local_accuracy_history else 0
            avg_accuracy = np.mean(self.local_accuracy_history) if self.local_accuracy_history else 0
           
            print(f"\n📈 Learning Performance:")
            print(f"  Final accuracy: {final_accuracy:.4f}")
            print(f"  Best accuracy: {best_accuracy:.4f}")
            print(f"  Average accuracy: {avg_accuracy:.4f}")
           
            if successful_rounds > 1 and self.local_accuracy_history[0] > 0:
                improvement = ((final_accuracy - self.local_accuracy_history[0]) /
                             self.local_accuracy_history[0]) * 100
                print(f"  Improvement: {improvement:+.1f}%")
       
        print(f"\n💾 All models saved in: device_{self.device_id}_models/")
        print(f"📊 Results saved in: device_{self.device_id}_results/")
        print("🎉 Device participation completed!")


# ==================== MAIN ====================
def main():
    device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
   
    print(f"🚀 Starting Device {device_id} with REAL Fashion-MNIST...")
   
    # Check if gRPC modules exist
    try:
        import federated_learning_pb2
        import federated_learning_pb2_grpc
    except ImportError:
        print("❌ gRPC modules not found!")
        print("Make sure to generate protobuf files first:")
        print("python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. federated_learning.proto")
        return
   
    # Check if PyTorch and torchvision are installed
    try:
        import torch
        import torchvision
        print(f"✅ PyTorch {torch.__version__}, torchvision {torchvision.__version__}")
    except ImportError:
        print("❌ PyTorch/torchvision not found!")
        print("Install with: pip install torch torchvision matplotlib")
        return
   
    # Create and run client
    client = DeviceFLClient(device_id)
   
    try:
        client.run_experiment()
    except KeyboardInterrupt:
        print(f"\n⚠️ Device {device_id}: Experiment stopped by user")
    except Exception as e:
        print(f"\n❌ Device {device_id}: Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()



   
    # Create and run client
    client = DeviceFLClient(device_id)
   
    try:
        client.run_experiment()
    except KeyboardInterrupt:
        print(f"\n⚠️ Device {device_id}: Experiment stopped by user")
    except Exception as e:
        print(f"\n❌ Device {device_id}: Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

