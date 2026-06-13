import grpc
from concurrent import futures
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import pickle
import json
import os
import subprocess
import sys
from datetime import datetime
import socket
from colorama import init, Fore, Style


init()  # Initialize colorama


import federated_learning_pb2
import federated_learning_pb2_grpc

import signal
import sys

import signal
import sys

# Define the handler function
def signal_handler(sig, frame):
    print("\n\n⚠️  Ctrl+C detected! Saving current results...")
    
    # Try to save current state
    try:
        # Get cloud_service from global scope if it exists
        cloud_service_obj = globals().get('cloud_service')
        if cloud_service_obj and hasattr(cloud_service_obj, 'save_incremental_results'):
            current_round = cloud_service_obj.current_round if hasattr(cloud_service_obj, 'current_round') else 0
            cloud_service_obj.save_incremental_results(current_round)
            print_success(f"Results saved up to round {current_round}")
    except Exception as e:
        print_error(f"Error saving results: {e}")
        # Try to save basic results anyway
        try:
            save_basic_results()
        except:
            pass
    
    print("👋 Exiting gracefully...")
    sys.exit(0)

# Helper function to save basic results if cloud_service isn't available
def save_basic_results():
    """Save basic results if cloud_service isn't available"""
    try:
        os.makedirs('results', exist_ok=True)
        
        # Helper function to convert numpy/pytorch types to Python native types
        def convert_to_python_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            else:
                return obj
        
        basic_data = {
            'rewards': experiment_results['rewards'],
            'fl_accuracies': experiment_results['fl_accuracies'],
            'fl_losses': experiment_results['fl_losses'],
            'training_times': experiment_results['training_times'],
            'rounds_completed': len(experiment_results['rewards']),
            'last_saved': datetime.now().isoformat()
        }
        
        # Convert all numpy/pytorch types to Python native types
        basic_data = convert_to_python_types(basic_data)
        
        with open('results/emergency_save.json', 'w') as f:
            json.dump(basic_data, f, indent=2)
        print_success("Emergency results saved")
    except Exception as e:
        print_error(f"Failed to save emergency results: {e}")

# Register signal handler at module load
signal.signal(signal.SIGINT, signal_handler)

# ==================== CONFIGURATION ====================
# NOTE: Update these IPs to match your network environment
CLOUD_IP = '0.0.0.0'           # Cloud listens on all interfaces
CLOUD_PORT = 5000
EXTERNAL_IP = '<CLOUD_VM_IP>'  # Replace with your Cloud VM's IP address (e.g., '192.168.x.x')


# Edge configurations - Update 'internal_ip' for each edge VM in your setup
# Each edge VM runs edge.py and communicates with the cloud on port 5001
EDGE_CONFIGS = [
    {'internal_ip': '<EDGE_VM_1_IP>', 'external_ip': EXTERNAL_IP, 'internal_port': 5001, 'ram_gb': 4, 'cores': 8, 'storage_gb': 80, 'status': 'unknown'},
    {'internal_ip': '<EDGE_VM_2_IP>', 'external_ip': EXTERNAL_IP, 'internal_port': 5001, 'ram_gb': 8, 'cores': 8, 'storage_gb': 80, 'status': 'unknown'},
    {'internal_ip': '<EDGE_VM_3_IP>', 'external_ip': EXTERNAL_IP, 'internal_port': 5001, 'ram_gb': 8, 'cores': 8, 'storage_gb': 80, 'status': 'unknown'},
    # Add more edge VMs as needed following the same format:
    # {'internal_ip': '<EDGE_VM_N_IP>', 'external_ip': EXTERNAL_IP, 'internal_port': 5001, 'ram_gb': <RAM>, 'cores': <CORES>, 'storage_gb': <STORAGE>, 'status': 'unknown'},
]


NUM_EDGES = len(EDGE_CONFIGS)
NUM_DEVICES = 3  # Number of device clients participating in federated learning


# ==================== DDQN PARAMETERS ====================
STATE_SIZE = NUM_EDGES * 6  # CPU, MEM, BW, Latency, Stress, Availability
ACTION_SIZE = NUM_EDGES
ROUNDS = 50
BATCH_SIZE = 32
MEMORY_SIZE = 2000
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
TARGET_UPDATE = 5
WARMUP_ROUNDS = 5


# ILP Weight Coefficients (λ) - Following your mathematical model
# λ₁ for Latency, λ₂ for Energy, λ₃ for CPU
# Note: In your setup, we use Stress instead of Energy
LAMBDA = [0.4, 0.3, 0.3]  # λ₁=0.4, λ₂=0.3, λ₃=0.3


# ==================== DDQN NETWORK ====================
class DDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
   
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# ==================== FL MODEL ====================
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
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


# ==================== GLOBAL STATE ====================
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
replay_buffer = deque(maxlen=MEMORY_SIZE)


policy_net = DDQN(STATE_SIZE, ACTION_SIZE)
target_net = DDQN(STATE_SIZE, ACTION_SIZE)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)


global_model = FashionMNISTModel()
global_weights = [param.data.clone() for param in global_model.parameters()]


STRESS_MAP = {0: 'NO_STRESS', 1: 'LOW_STRESS', 2: 'MEDIUM_STRESS', 3: 'HIGH_STRESS'}


edge_stress_levels = {i: 0 for i in range(NUM_EDGES)}
edge_metrics = {}
device_registry = {}
current_round = 0
epsilon = EPSILON_START
edge_availability = {i: False for i in range(NUM_EDGES)}


experiment_results = {
    'rounds': [], 'rewards': [], 'fl_accuracies': [], 'fl_losses': [],
    'training_times': [], 'assignments': [], 'epsilon_values': [],
    'ddqn_decisions': [], 'ilp_decisions': [], 'final_decisions': [],
    'detailed_edge_logs': []
}


# ==================== LOGGING UTILITIES ====================
def print_header(text):
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")


def print_subheader(text):
    print(f"\n{Fore.GREEN}{text}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'-'*len(text)}{Style.RESET_ALL}")


def print_success(text):
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_warning(text):
    print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")


def print_error(text):
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def print_info(text):
    print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")


def print_metric(label, value, unit=""):
    print(f"{Fore.MAGENTA}  {label}: {Fore.WHITE}{value}{unit}{Style.RESET_ALL}")


# ==================== EDGE MANAGEMENT ====================
def check_edge_availability(edge_idx):
    """Check if edge is actually reachable"""
    edge_cfg = EDGE_CONFIGS[edge_idx]
   
    try:
        channel = grpc.insecure_channel(f"{edge_cfg['internal_ip']}:{edge_cfg['internal_port']}")
        stub = federated_learning_pb2_grpc.EdgeNodeStub(channel)
        response = stub.GetMetrics(federated_learning_pb2.MetricsRequest(), timeout=3)
       
        if response:
            edge_availability[edge_idx] = True
            EDGE_CONFIGS[edge_idx]['status'] = 'active'
            return True
    except:
        pass
   
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((edge_cfg['internal_ip'], edge_cfg['internal_port']))
        sock.close()
       
        if result == 0:
            edge_availability[edge_idx] = True
            EDGE_CONFIGS[edge_idx]['status'] = 'active'
            return True
    except:
        pass
   
    edge_availability[edge_idx] = False
    EDGE_CONFIGS[edge_idx]['status'] = 'inactive'
    return False


def apply_stress_to_edge(edge_idx, stress_level):
    """Apply stress to edge via gRPC with proper enum handling"""
    if not edge_availability[edge_idx]:
        return False
   
    edge_cfg = EDGE_CONFIGS[edge_idx]
   
    try:
        channel = grpc.insecure_channel(f"{edge_cfg['internal_ip']}:{edge_cfg['internal_port']}")
        stub = federated_learning_pb2_grpc.EdgeNodeStub(channel)
       
        stress_enum_map = {
            0: federated_learning_pb2.NO_STRESS,
            1: federated_learning_pb2.LOW_STRESS,
            2: federated_learning_pb2.MEDIUM_STRESS,
            3: federated_learning_pb2.HIGH_STRESS
        }
       
        stress_enum = stress_enum_map.get(stress_level, federated_learning_pb2.NO_STRESS)
       
        response = stub.ApplyStress(
            federated_learning_pb2.StressRequest(
                level=stress_enum,
                duration_seconds=60
            ),
            timeout=5
        )
       
        if response.applied:
            edge_stress_levels[edge_idx] = stress_level
            stress_name = STRESS_MAP[stress_level]
            print_success(f"Edge {edge_idx} ({edge_cfg['internal_ip']}): Applied {stress_name}")
            return True
        else:
            print_warning(f"Edge {edge_idx}: Stress application failed: {response.message}")
            return False
    except Exception as e:
        print_warning(f"Edge {edge_idx}: Cannot apply stress - {e}")
        return False


def get_edge_metrics(edge_idx):
    """Get metrics from edge with proper enum handling"""
    edge_cfg = EDGE_CONFIGS[edge_idx]
   
    if not edge_availability[edge_idx]:
        return {
            'cpu': 100.0, 'memory': 100.0, 'bandwidth': 0.0,
            'latency': 999.0, 'stress': edge_stress_levels[edge_idx],
            'available': 0.0
        }
   
    try:
        channel = grpc.insecure_channel(f"{edge_cfg['internal_ip']}:{edge_cfg['internal_port']}")
        stub = federated_learning_pb2_grpc.EdgeNodeStub(channel)
       
        response = stub.GetMetrics(federated_learning_pb2.MetricsRequest(), timeout=3)
       
        if response:
            stress_numeric = {
                federated_learning_pb2.NO_STRESS: 0,
                federated_learning_pb2.LOW_STRESS: 1,
                federated_learning_pb2.MEDIUM_STRESS: 2,
                federated_learning_pb2.HIGH_STRESS: 3
            }.get(response.stress_level, 0)
           
            return {
                'cpu': response.cpu_usage,
                'memory': response.memory_usage,
                'bandwidth': response.bandwidth,
                'latency': response.latency,
                'stress': stress_numeric,
                'available': 1.0
            }
    except Exception as e:
        print_warning(f"Edge {edge_idx}: Metrics error - {e}")
   
    return {
        'cpu': 50.0, 'memory': 50.0, 'bandwidth': 10.0,
        'latency': 100.0, 'stress': edge_stress_levels[edge_idx],
        'available': 0.5
    }


def get_system_state():
    """Get complete system state with centralized logging"""
    print_subheader("📊 COLLECTING REAL-TIME METRICS FROM ALL EDGES")
   
    state = []
    active_edges = 0
   
    for i in range(NUM_EDGES):
        is_available = check_edge_availability(i)
        edge_cfg = EDGE_CONFIGS[i]
       
        if is_available:
            metrics = get_edge_metrics(i)
            active_edges += 1
        else:
            metrics = {
                'cpu': 100.0, 'memory': 100.0, 'bandwidth': 0.0,
                'latency': 999.0, 'stress': edge_stress_levels[i],
                'available': 0.0
            }
       
        # Normalize values (0-1 scale as per mathematical model)
        cpu_norm = min(metrics['cpu'] / 100.0, 1.0)
        mem_norm = min(metrics['memory'] / 100.0, 1.0)
        bw_norm = min(metrics['bandwidth'] / 100.0, 1.0)
        latency_norm = min(metrics['latency'] / 500.0, 1.0)
        stress_norm = metrics['stress'] / 3.0  # Using stress instead of energy
        avail_norm = metrics['available']
       
        state.extend([cpu_norm, mem_norm, bw_norm, latency_norm, stress_norm, avail_norm])
       
        edge_metrics[i] = metrics
       
        status_color = Fore.GREEN if is_available else Fore.RED
        status_text = "ACTIVE" if is_available else "INACTIVE"
       
        print(f"{status_color}┌ Edge {i:2d} ({edge_cfg['internal_ip']}): {status_text}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}│   CPU: {metrics['cpu']:6.1f}% | Memory: {metrics['memory']:6.1f}% | BW: {metrics['bandwidth']:6.1f} Mbps")
        print(f"{Fore.WHITE}│   Latency: {metrics['latency']:6.1f} ms | Stress: {STRESS_MAP.get(metrics['stress'], 'UNKNOWN'):15s}{Style.RESET_ALL}")
   
    print_info(f"Active edges: {active_edges}/{NUM_EDGES}")
    return np.array(state, dtype=np.float32)


# ==================== ILP OPTIMIZATION ====================
def ilp_optimization(state, round_num, available_edges):
    """ILP optimization following mathematical model: Minimize Z = Σ(λ₁T̃ + λ₂Ẽ + λ₃CPŪ)"""
    print_subheader("🔍 ILP OPTIMIZATION")
    print_metric("Round", round_num)
    print_metric("Lambda weights", f"λ₁(Latency)={LAMBDA[0]}, λ₂(Stress)={LAMBDA[1]}, λ₃(CPU)={LAMBDA[2]}")
   
    if not available_edges:
        print_error("No available edges for ILP optimization")
        return -1, float('inf')
   
    best_edge = -1
    best_cost = float('inf')
   
    print_info(f"Analyzing {len(available_edges)} available edges:")
   
    for edge_idx in available_edges:
        base_idx = edge_idx * 6
       
        # Extract normalized metrics (already 0-1 scale)
        cpu_norm = state[base_idx]      # CPŪ_{t,j}
        mem_norm = state[base_idx + 1]
        bw_norm = state[base_idx + 2]
        latency_norm = state[base_idx + 3]  # T̃_{t,j}
        stress_norm = state[base_idx + 4]   # Ẽ_{t,j} (using stress as energy proxy)
        avail_norm = state[base_idx + 5]
       
        # Skip if not available
        if avail_norm < 0.5:
            continue
       
        # ILP Objective Function: Z = λ₁*T̃ + λ₂*Ẽ + λ₃*CPŪ
        # Following: Z_t = Σ_{j=1}^M (λ₁T̃_{t,j} + λ₂Ẽ_{t,j} + λ₃CPŪ_{t,j}) * x_{t,j}
        total_cost = (LAMBDA[0] * latency_norm +    # λ₁ * T̃ (Latency)
                     LAMBDA[1] * stress_norm +      # λ₂ * Ẽ (Stress/Energy)
                     LAMBDA[2] * cpu_norm)          # λ₃ * CPŪ (CPU)
       
        # Resource constraints (ILP constraints)
        edge_cfg = EDGE_CONFIGS[edge_idx]
        cpu_capacity = edge_cfg['cores'] * 100
        mem_capacity = edge_cfg['ram_gb'] * 1024
       
        # Task requirements for FL training
        task_cpu = 200
        task_mem = 512
        task_bw = 10
       
        # Check feasibility
        current_cpu = cpu_norm * 100
        current_mem = mem_norm * 100
       
        # Constraint: CPU_{t,j} + CPU_task ≤ CPU_max
        cpu_feasible = (current_cpu + (task_cpu / cpu_capacity * 100)) <= 90
       
        # Constraint: MEM_{t,j} + MEM_task ≤ MEM_max
        mem_feasible = (current_mem + (task_mem / mem_capacity * 100)) <= 90
       
        # Constraint: BW_{t,j} + BW_task ≤ BW_max
        bw_feasible = bw_norm <= 0.8
       
        feasible = cpu_feasible and mem_feasible and bw_feasible
       
        if feasible and total_cost < best_cost:
            best_cost = total_cost
            best_edge = edge_idx
       
        # Detailed logging
        status = "✓ FEASIBLE" if feasible else "✗ INFEASIBLE"
        color = Fore.GREEN if feasible else Fore.RED
       
        print(f"{color}  Edge {edge_idx:2d}: {status}")
        print(f"    Cost (Z) = {total_cost:.4f} = {LAMBDA[0]:.1f}×{latency_norm:.3f} + {LAMBDA[1]:.1f}×{stress_norm:.3f} + {LAMBDA[2]:.1f}×{cpu_norm:.3f}")
        if not feasible:
            print(f"    Constraints: CPU={cpu_feasible}, Memory={mem_feasible}, BW={bw_feasible}")
        print(f"{Style.RESET_ALL}", end="")
   
    if best_edge != -1:
        print_success(f"ILP Solution: Edge {best_edge} with minimum cost Z={best_cost:.4f}")
        return best_edge, best_cost
    else:
        print_warning("ILP: No feasible solution found")
        return -1, float('inf')


# ==================== DDQN DECISION ====================
def ddqn_select_action(state_tensor, round_num, available_edges):
    """DDQN action selection following mathematical model: a_t = argmax_a Q(S_t, a; Θ)"""
    global epsilon
   
    print_subheader("🤖 DDQN AGENT DECISION")
    print_metric("Round", round_num)
    print_metric("Epsilon", f"{epsilon:.3f}")
    print_metric("Available edges", f"{len(available_edges)}/{NUM_EDGES}")
   
    # Exploration vs Exploitation
    if random.random() < epsilon:
        # Exploration: randomly select from available edges
        if available_edges:
            action = random.choice(available_edges)
            print_success(f"Exploration: Randomly selected Edge {action}")
            return action
        else:
            print_error("No available edges for exploration")
            return -1
    else:
        # Exploitation: use DDQN to select best edge
        # Following: a_t = argmax_a Q(S_t, a; Θ)
        with torch.no_grad():
            q_values = policy_net(state_tensor).numpy().flatten()
       
        # Set unavailable edges to -inf
        for i in range(NUM_EDGES):
            if i not in available_edges:
                q_values[i] = -float('inf')
       
        if np.all(q_values == -float('inf')):
            print_error("All edges have negative infinite Q-values")
            return -1
       
        action = np.argmax(q_values)
        q_val = q_values[action]
       
        print_success(f"Exploitation: Selected Edge {action} with Q-value = {q_val:.4f}")
       
        # Log top 5 Q-values
        print_info("Top 5 Q-values:")
        sorted_indices = np.argsort(q_values)[-5:][::-1]
        for idx in sorted_indices:
            if q_values[idx] > -float('inf'):
                status = "✓" if idx in available_edges else "✗"
                print(f"  {status} Edge {idx:2d}: Q = {q_values[idx]:8.4f}")
       
        return action


# ==================== HYBRID DECISION ====================
def hybrid_ddqn_ilp_decision(state, round_num):
    """Hybrid DDQN+ILP decision following mathematical model"""
    print_header(f"🚀 HYBRID DDQN+ILP DECISION - ROUND {round_num}")
   
    # Get available edges
    available_edges = [i for i in range(NUM_EDGES) if edge_availability[i]]
   
    if not available_edges:
        print_error("❌ CRITICAL: No edges available!")
        return -1
   
    # === PHASE 1: WARMUP (ROUNDS 1-5) - ILP ONLY ===
    if round_num <= WARMUP_ROUNDS:
        print_subheader(f"🔥 WARMUP PHASE {round_num}/{WARMUP_ROUNDS} - ILP ONLY")
        
        ilp_edge, ilp_cost = ilp_optimization(state, round_num, available_edges)
        
        if ilp_edge != -1:
            print_success(f"Warmup: ILP selects Edge {ilp_edge} with cost Z={ilp_cost:.4f}")
            
            # Store ILP decision for warmup
            experiment_results['ddqn_decisions'].append({
                'round': round_num,
                'type': 'ILP_WARMUP',
                'edge': ilp_edge,
                'cost': ilp_cost,
                'q_value': None
            })
            
            # ADD THIS: Store separate ILP decision
            experiment_results['ilp_decisions'].append({
                'round': round_num,
                'ilp_edge': ilp_edge,
                'ilp_cost': float(ilp_cost),
                'feasibility': True,
                'phase': 'warmup'
            })
            
            return ilp_edge
        else:
            # Fallback to random selection if ILP fails
            print_warning("ILP failed during warmup, using random selection")
            if available_edges:
                random_edge = random.choice(available_edges)
                print_success(f"Random fallback to Edge {random_edge}")
                
                # ADD THIS: Store failed ILP decision
                experiment_results['ilp_decisions'].append({
                    'round': round_num,
                    'ilp_edge': -1,
                    'ilp_cost': None,
                    'feasibility': False,
                    'phase': 'warmup'
                })
                
                return random_edge
   
    # === PHASE 2: HYBRID (ROUNDS 6-50) ===
    print_subheader("🎯 HYBRID DDQN+ILP DECISION")
   
    # Step 1: DDQN Decision
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    ddqn_edge = ddqn_select_action(state_tensor, round_num, available_edges)
   
    # Get DDQN Q-value
    with torch.no_grad():
        q_values = policy_net(state_tensor).numpy().flatten()
    ddqn_q_value = q_values[ddqn_edge] if ddqn_edge != -1 else -float('inf')
   
    # Step 2: ILP Optimization
    ilp_edge, ilp_cost = ilp_optimization(state, round_num, available_edges)
   
    # Step 3: Calculate DDQN's cost for comparison
    if ddqn_edge != -1:
        base_idx = ddqn_edge * 6
        ddqn_latency = state[base_idx + 3]
        ddqn_stress = state[base_idx + 4]
        ddqn_cpu = state[base_idx]
       
        ddqn_cost = (LAMBDA[0] * ddqn_latency +
                    LAMBDA[1] * ddqn_stress +
                    LAMBDA[2] * ddqn_cpu)
    else:
        ddqn_cost = float('inf')
   
    # Store both decisions
    experiment_results['ddqn_decisions'].append({
        'round': round_num,
        'ddqn_edge': ddqn_edge,
        'ddqn_q_value': float(ddqn_q_value) if ddqn_edge != -1 else None,
        'ddqn_cost': float(ddqn_cost) if ddqn_edge != -1 else None,
        'ilp_edge': ilp_edge,
        'ilp_cost': float(ilp_cost) if ilp_edge != -1 else None
    })
    experiment_results['ilp_decisions'].append({
    'round': round_num,
    'ilp_edge': ilp_edge,
    'ilp_cost': float(ilp_cost) if ilp_edge != -1 else None,
    'feasibility': ilp_edge != -1,
    'phase': 'hybrid'
})
   
    # === DECISION LOGIC ===
    # Case 1: Both algorithms agree
    if ddqn_edge == ilp_edge and ddqn_edge != -1:
        final_edge = ddqn_edge
        decision_reason = "Consensus: DDQN and ILP agree"
   
    # Case 2: DDQN failed, ILP succeeded
    elif ddqn_edge == -1 and ilp_edge != -1:
        final_edge = ilp_edge
        decision_reason = "ILP selected (DDQN failed)"
   
    # Case 3: ILP failed, DDQN succeeded
    elif ilp_edge == -1 and ddqn_edge != -1:
        final_edge = ddqn_edge
        decision_reason = "DDQN selected (ILP failed)"
   
    # Case 4: Both succeeded but disagree - use hybrid logic
    elif ddqn_edge != -1 and ilp_edge != -1 and ddqn_edge != ilp_edge:
        cost_difference = ddqn_cost - ilp_cost
       
        print_info("Hybrid Analysis:")
        print_metric(f"DDQN Edge {ddqn_edge}", f"Q={ddqn_q_value:.4f}, Cost={ddqn_cost:.4f}")
        print_metric(f"ILP Edge {ilp_edge}", f"Cost={ilp_cost:.4f}")
        print_metric("Cost Difference", f"{cost_difference:.4f}")
       
        # Decision rule:
        # 1. If ILP is significantly better (>10% cost reduction), use ILP
        # 2. If DDQN is very confident (Q > threshold), use DDQN
        # 3. Otherwise, use ILP (more conservative)
       
        if cost_difference > 0.1:  # ILP is 10% better
            final_edge = ilp_edge
            decision_reason = f"ILP better (cost diff: {cost_difference:.4f})"
        elif ddqn_q_value > 0.5:  # DDQN is confident
            final_edge = ddqn_edge
            decision_reason = f"DDQN confident (Q={ddqn_q_value:.4f})"
        else:
            final_edge = ilp_edge
            decision_reason = f"Conservative: ILP (DDQN Q={ddqn_q_value:.4f})"
   
    # Case 5: Both failed
    else:
        print_error("Both DDQN and ILP failed")
        if available_edges:
            final_edge = random.choice(available_edges)
            decision_reason = "Random fallback (both algorithms failed)"
        else:
            print_error("No available edges at all!")
            return -1
   
    # Final logging
    print_header(f"🎯 FINAL DECISION - ROUND {round_num}")
    print_metric("DDQN", f"Edge {ddqn_edge} (Q={ddqn_q_value:.4f}, Cost={ddqn_cost:.4f})" if ddqn_edge != -1 else "Failed")
    print_metric("ILP", f"Edge {ilp_edge} (Cost={ilp_cost:.4f})" if ilp_edge != -1 else "Failed")
    print_metric("Final Selection", f"Edge {final_edge}")
    print_metric("Decision Reason", decision_reason)
   
    # Edge details
    if final_edge != -1:
        edge_cfg = EDGE_CONFIGS[final_edge]
        metrics = edge_metrics.get(final_edge, {})
        print_info("Selected Edge Details:")
        print_metric("IP Address", edge_cfg['internal_ip'])
        print_metric("CPU Usage", f"{metrics.get('cpu', 0):.1f}%")
        print_metric("Latency", f"{metrics.get('latency', 0):.1f} ms")
        print_metric("Stress Level", STRESS_MAP.get(metrics.get('stress', 0), "UNKNOWN"))
   
    # Store final decision
    experiment_results['final_decisions'].append({
        'round': round_num,
        'ddqn_edge': ddqn_edge,
        'ilp_edge': ilp_edge,
        'final_edge': final_edge,
        'decision_reason': decision_reason,
        'ddqn_q_value': float(ddqn_q_value) if ddqn_edge != -1 else None,
        'ddqn_cost': float(ddqn_cost) if ddqn_edge != -1 else None,
        'ilp_cost': float(ilp_cost) if ilp_edge != -1 else None
    })
   
    return final_edge


# ==================== REWARD CALCULATION ====================
def calculate_reward(state, edge_idx, training_time, accuracy, round_num):
    """Calculate reward following mathematical model: R_t = -(λ₁Latency + λ₂Energy + λ₃CPU)"""
    if edge_idx == -1:
        return -10.0
   
    base_idx = edge_idx * 6
    cpu_norm = state[base_idx]          # CPŪ
    latency_norm = state[base_idx + 3]  # T̃ (Latency)
    stress_norm = state[base_idx + 4]   # Ẽ (Stress/Energy)
   
    # Following: R_t = -(λ₁Latency_t + λ₂Energy_t + λ₃CPU_t)
    # Using normalized values (0-1 scale)
    latency_component = LAMBDA[0] * latency_norm
    stress_component = LAMBDA[1] * stress_norm
    cpu_component = LAMBDA[2] * cpu_norm
   
    total_cost = latency_component + stress_component + cpu_component
   
    # Negative reward (penalty) as per mathematical model
    reward = -total_cost
   
    # Add bonus for accuracy and time efficiency
    accuracy_bonus = accuracy * 0.5
    time_bonus = max(0, 1.0 - (training_time / 120.0))  # 2 minute reference
   
    final_reward = reward + accuracy_bonus + time_bonus
   
    print_subheader(f"🎁 REWARD CALCULATION - ROUND {round_num}")
    print_metric("Selected Edge", edge_idx)
    print_info("Mathematical Model Components:")
    print_metric("  λ₁×Latency", f"{latency_component:.4f} = {LAMBDA[0]}×{latency_norm:.3f}")
    print_metric("  λ₂×Stress", f"{stress_component:.4f} = {LAMBDA[1]}×{stress_norm:.3f}")
    print_metric("  λ₃×CPU", f"{cpu_component:.4f} = {LAMBDA[2]}×{cpu_norm:.3f}")
    print_metric("  Total Cost (Z)", f"{total_cost:.4f}")
    print_metric("  Base Reward", f"{reward:.4f}")
    print_info("Additional Bonuses:")
    print_metric("  Accuracy Bonus", f"{accuracy_bonus:.4f}")
    print_metric("  Time Bonus", f"{time_bonus:.4f}")
    print_metric("Final Reward (R_t)", f"{final_reward:.4f}")
   
    return final_reward


# ==================== DDQN TRAINING ====================
def train_ddqn():
    """Train DDQN using Double DQN algorithm following mathematical model"""
    if len(replay_buffer) < BATCH_SIZE:
        return None
   
    batch = random.sample(replay_buffer, BATCH_SIZE)
    batch_t = Transition(*zip(*batch))
   
    state_batch = torch.FloatTensor(np.array(batch_t.state))
    action_batch = torch.LongTensor(batch_t.action)
    reward_batch = torch.FloatTensor(batch_t.reward)
    next_state_batch = torch.FloatTensor(np.array(batch_t.next_state))
    done_batch = torch.FloatTensor(batch_t.done)
   
    # Current Q values: Q(S_t, a_t; Θ)
    current_q = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
   
    # Double DQN target calculation:
    # Y_t = R_t + γ * Q'(S_{t+1}, argmax_a Q(S_{t+1}, a; Θ); Θ')
    with torch.no_grad():
        # Next actions from online network
        next_actions = policy_net(next_state_batch).max(1)[1].unsqueeze(1)
       
        # Q values from target network for those actions
        next_q = target_net(next_state_batch).gather(1, next_actions).squeeze()
       
        # Target Q values
        target_q = reward_batch + GAMMA * next_q * (1.0 - done_batch)
   
    # Loss: L(Θ) = (Y_t - Q(S_t, a_t; Θ))²
    loss = nn.MSELoss()(current_q, target_q)
   
    # Optimize: Θ ← Θ - α ∇_Θ L(Θ)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
   
    return loss.item()


# ==================== FL TRAINING MANAGEMENT ====================
def collect_fl_results_from_edge(edge_idx, round_num):
    """Collect FL results from edge"""
    if not edge_availability[edge_idx]:
        return None
   
    edge_cfg = EDGE_CONFIGS[edge_idx]
   
    try:
        print_subheader(f"📥 COLLECTING FL RESULTS FROM EDGE {edge_idx}")
       
        channel = grpc.insecure_channel(f"{edge_cfg['internal_ip']}:{edge_cfg['internal_port']}")
        stub = federated_learning_pb2_grpc.EdgeNodeStub(channel)
       
        response = stub.GetTrainingResults(
            federated_learning_pb2.ResultsRequest(),
            timeout=15
        )
       
        if response.model_weights:
            weights = pickle.loads(response.model_weights)
           
            result = {
                'weights': weights,
                'accuracy': response.accuracy,
                'loss': response.loss,
                'training_time': response.training_time,
                'samples': response.samples_trained,
                'edge_id': edge_idx,
                'round': round_num
            }
           
            print_success(f"Edge {edge_idx}: FL Accuracy={response.accuracy:.4f}, Loss={response.loss:.4f}, "
                         f"Time={response.training_time}s, Samples={response.samples_trained}")
           
            return result
        else:
            print_warning(f"Edge {edge_idx}: No valid results returned")
            return None
           
    except Exception as e:
        print_error(f"Edge {edge_idx}: Failed to collect results - {e}")
        return None


def federated_average(edge_results):
    """Federated averaging"""
    if not edge_results:
        print_error("No edge results for federated averaging")
        return None
   
    print_subheader("🔄 FEDERATED AVERAGING")
   
    total_samples = sum(r['samples'] for r in edge_results)
    averaged_weights = []
   
    print_info(f"Averaging from {len(edge_results)} edges")
    print_info(f"Total samples: {total_samples}")
   
    for i in range(len(edge_results[0]['weights'])):
        weighted_sum = None
        for result in edge_results:
            weighted_weights = result['weights'][i] * result['samples']
            if weighted_sum is None:
                weighted_sum = weighted_weights
            else:
                weighted_sum += weighted_weights
       
        if total_samples > 0:
            averaged_weights.append(weighted_sum / total_samples)
   
    avg_accuracy = np.mean([r['accuracy'] for r in edge_results])
    avg_loss = np.mean([r['loss'] for r in edge_results])
    avg_time = np.mean([r['training_time'] for r in edge_results])
   
    print_success(f"Federated Averaging Complete:")
    print_metric("Average Accuracy", f"{avg_accuracy:.4f}")
    print_metric("Average Loss", f"{avg_loss:.4f}")
    print_metric("Average Training Time", f"{avg_time:.1f} s")
    print_metric("Total Samples", f"{total_samples}")
   
    return averaged_weights, avg_accuracy, avg_loss, avg_time


# ==================== gRPC SERVICE ====================
class CloudCoreService(federated_learning_pb2_grpc.CloudCoreServicer):
    def __init__(self):
        self.current_round = 0
        self.round_active = False
        self.device_assignments = {}
        self.ready_devices = set()
        self.edge_assignments = {}
        self.lock = threading.Lock()
       
        print_header("☁️ CLOUD CORE SERVICE INITIALIZED")
        print_metric("Cloud IP", f"{EXTERNAL_IP}:{CLOUD_PORT}")
        print_metric("Edge VMs", f"{NUM_EDGES}")
        print_metric("Devices", f"{NUM_DEVICES}")
        print_metric("DDQN+ILP", "Enabled")
        print_metric("Warmup Rounds", f"{WARMUP_ROUNDS}")
        print_metric("Lambda Weights", f"λ={LAMBDA}")
   
    def RegisterDevice(self, request, context):
        with self.lock:
            device_registry[request.device_id] = {
                'ip': request.ip_address,
                'samples': request.data_samples,
                'last_seen': time.time()
            }
           
            print_success(f"Device {request.device_id} registered from {request.ip_address}")
           
            return federated_learning_pb2.RegistrationResponse(
                success=True,
                message=f"Device {request.device_id} registered"
            )
   
    def ReadyForRound(self, request, context):
        with self.lock:
            print_info(f"Device {request.device_id} ready for round {request.round}")
           
            if not self.round_active or request.round != self.current_round:
                return federated_learning_pb2.ReadyResponse(
                    acknowledged=False,
                    wait=True
                )
           
            self.ready_devices.add(request.device_id)
           
            print_metric("Ready devices", f"{len(self.ready_devices)}/{NUM_DEVICES}")
           
            wait_needed = (len(self.ready_devices) < NUM_DEVICES)
           
            return federated_learning_pb2.ReadyResponse(
                acknowledged=True,
                wait=wait_needed
            )
   
    def GetAssignment(self, request, context):
        """Handle device assignment request - FIXED VERSION"""
        with self.lock:
            device_id = request.device_id
            requested_round = request.round
           
            print(f"\n📋 Device {device_id} requesting assignment for round {requested_round}")
            print(f"   Cloud state: current_round={self.current_round}, round_active={self.round_active}")
           
            # Check if device is ready
            if device_id not in self.ready_devices:
                print(f"   ❌ Device {device_id} not in ready_devices set")
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("Device not ready. Call ReadyForRound first.")
                return federated_learning_pb2.AssignmentResponse()
           
            # Check round synchronization
            if not self.round_active:
                print(f"   ❌ Round not active")
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("Round not active. Wait for cloud to start round.")
                return federated_learning_pb2.AssignmentResponse()
           
            if requested_round != self.current_round:
                print(f"   ❌ Round mismatch: device={requested_round}, cloud={self.current_round}")
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(f"Round mismatch. Device requested round {requested_round}, but cloud is on round {self.current_round}")
                return federated_learning_pb2.AssignmentResponse()
           
            # Already assigned? Return existing assignment
            if device_id in self.device_assignments:
                edge_idx = self.device_assignments[device_id]
                edge_cfg = EDGE_CONFIGS[edge_idx]
                print(f"   ✅ Already assigned: Device {device_id} → Edge {edge_idx}")
               
                return federated_learning_pb2.AssignmentResponse(
                    edge_ip=edge_cfg['external_ip'],
                    edge_port=5001,
                    edge_id=edge_idx,
                    round=self.current_round
                )
           
            # ========== NEW ASSIGNMENT LOGIC ==========
            print(f"\n🎯 Making NEW assignment for Device {device_id} (Round {self.current_round})")
           
            # Step 1: Get system state
            print("   📊 Collecting system state...")
            state = get_system_state()
           
            # Step 2: Apply stress (with probability)
            if random.random() < 0.3:  # 30% chance to apply stress
                active_edges = [i for i in range(NUM_EDGES) if edge_availability[i]]
                if active_edges:
                    edge_idx = random.choice(active_edges)
                    stress_level = random.choices([0, 1, 2, 3], weights=[0.3, 0.3, 0.2, 0.2])[0]
                   
                    if stress_level > 0:
                        success = apply_stress_to_edge(edge_idx, stress_level)
                        if success:
                            print(f"   🔥 Applied {STRESS_MAP[stress_level]} to Edge {edge_idx}")
           
            # Step 3: Get hybrid decision
            print("   🤖 Running hybrid DDQN+ILP decision...")
            edge_idx = hybrid_ddqn_ilp_decision(state, self.current_round)
           
            if edge_idx == -1:
                print("   ❌ No suitable edge available")
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("No suitable edge available. Try again later.")
                return federated_learning_pb2.AssignmentResponse()
           
            # Step 4: Store assignment
            self.device_assignments[device_id] = edge_idx
           
            # Track edge assignments
            if edge_idx not in self.edge_assignments:
                self.edge_assignments[edge_idx] = []
            self.edge_assignments[edge_idx].append(device_id)
           
            edge_cfg = EDGE_CONFIGS[edge_idx]
           
            print(f"   ✅ Assigned: Device {device_id} → Edge {edge_idx} ({edge_cfg['internal_ip']})")
            print(f"   📍 Edge details: {edge_cfg['external_ip']}:5001, RAM={edge_cfg['ram_gb']}GB, Cores={edge_cfg['cores']}")
           
            return federated_learning_pb2.AssignmentResponse(
                edge_ip=edge_cfg['external_ip'],
                edge_port=5001,
                edge_id=edge_idx,
                round=self.current_round
            )
   
    def GetGlobalModel(self, request, context):
        return federated_learning_pb2.ModelResponse(
            model_weights=pickle.dumps(global_weights),
            round=self.current_round
        )
   
    def ReportMetrics(self, request, context):
        with self.lock:
            edge_id = request.edge_id
           
            stress_numeric = {
                federated_learning_pb2.NO_STRESS: 0,
                federated_learning_pb2.LOW_STRESS: 1,
                federated_learning_pb2.MEDIUM_STRESS: 2,
                federated_learning_pb2.HIGH_STRESS: 3
            }.get(request.stress_level, 0)
           
            edge_metrics[edge_id] = {
                'cpu': request.cpu_usage,
                'memory': request.memory_usage,
                'bandwidth': request.bandwidth,
                'latency': request.latency,
                'stress': stress_numeric
            }
           
            edge_stress_levels[edge_id] = stress_numeric
           
            return federated_learning_pb2.MetricsAck(received=True)
   
    def run_fl_round(self, round_num):
        """Run a federated learning round - IMPROVED TIMING"""
        global epsilon, global_weights
       
        print_header(f"🎯 ROUND {round_num}/{ROUNDS} - FEDERATED LEARNING")
       
        # Step 1: Reset state and open round
        with self.lock:
            self.current_round = round_num
            self.round_active = True
            self.ready_devices.clear()
            self.device_assignments.clear()
            self.edge_assignments.clear()
       
        print_info(f"📣 ROUND {round_num} OPENED")
        print_info(f"⏳ Waiting for {NUM_DEVICES} device(s) to signal readiness...")
       
        # Step 2: Wait for devices with better timing
        start_time = time.time()
        timeout_seconds = 60  # Give devices 60 seconds to connect
       
        while True:
            with self.lock:
                ready_count = len(self.ready_devices)
           
            print_metric("Devices ready", f"{ready_count}/{NUM_DEVICES}")
           
            if ready_count >= NUM_DEVICES:
                print_success(f"✅ All {NUM_DEVICES} device(s) ready!")
                break
           
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                print_error(f"⏰ Timeout: Only {ready_count}/{NUM_DEVICES} devices ready after {timeout_seconds}s")
                with self.lock:
                    self.round_active = False
                return False
           
            # Check every 2 seconds
            time.sleep(2)
       
        # Step 3: Get system state for decision making
        print_info("📊 Collecting system state for edge assignment...")
        state = get_system_state()
       
        # Step 4: Wait for device assignments
        print_info("⏳ Waiting for devices to request assignments (10 seconds)...")
        time.sleep(10)  # Give devices time to request assignments
       
        # Step 5: Check if any assignments were made
        with self.lock:
            if not self.device_assignments:
                print_warning("⚠️ No device requested assignment")
                self.round_active = False
                return False
       
        # Step 6: Wait for data offloading and training
        print_info("⏳ Waiting for data offloading and FL training (60 noooo seconds)...")
        # time.sleep(60)
        edge_results = []
       
        for edge_idx, devices in self.edge_assignments.items():
            print_info(f"Collecting results from Edge {edge_idx} (Devices: {devices})")
           
            result = collect_fl_results_from_edge(edge_idx, round_num)
            if result:
                edge_results.append(result)
       
        if edge_results:
            new_weights, avg_accuracy, avg_loss, avg_time = federated_average(edge_results)
           
            if new_weights:
                global_weights = new_weights
               
                if self.edge_assignments:
                    first_edge = list(self.edge_assignments.keys())[0]
                    reward = calculate_reward(state, first_edge, avg_time, avg_accuracy, round_num)
                else:
                    reward = -5.0
               
                experiment_results['rewards'].append(reward)
                experiment_results['fl_accuracies'].append(avg_accuracy)
                experiment_results['fl_losses'].append(avg_loss)
                experiment_results['training_times'].append(avg_time)
                experiment_results['assignments'].append(dict(self.device_assignments))
                experiment_results['epsilon_values'].append(epsilon)
               
                # DDQN Training (only after warmup)
                if round_num > WARMUP_ROUNDS and first_edge != -1:
                    next_state = get_system_state()
                   
                    # Store transition in replay buffer
                    replay_buffer.append(Transition(state, first_edge, reward, next_state, False))
                   
                    # Train DDQN
                    if len(replay_buffer) >= BATCH_SIZE:
                        loss = train_ddqn()
                        if loss is not None:
                            print_subheader("🤖 DDQN TRAINING UPDATE")
                            print_metric("Training Loss", f"{loss:.4f}")
                            print_metric("Replay Buffer Size", len(replay_buffer))
                   
                    # Update epsilon
                    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
                    print_metric("Updated Epsilon", f"{epsilon:.3f}")
                   
                    # Update target network
                    if round_num % TARGET_UPDATE == 0:
                        target_net.load_state_dict(policy_net.state_dict())
                        print_success(f"Target network updated (every {TARGET_UPDATE} rounds)")
               
                self._save_global_model(round_num, avg_accuracy, avg_loss)
       
        with self.lock:
            self.round_active = False
       
        print_success(f"Round {round_num} completed successfully")
        return True
   
    def _save_global_model(self, round_num, accuracy, loss):
        os.makedirs('saved_models', exist_ok=True)
       
        model_data = {
            'weights': global_weights,
            'round': round_num,
            'accuracy': accuracy,
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'epsilon': epsilon,
            'lambda_weights': LAMBDA
        }
       
        filename = f"saved_models/global_model_round_{round_num:03d}.pth"
        torch.save(model_data, filename)
        print_success(f"Model saved: {filename}")
   
    def save_experiment_results(self):
        os.makedirs('results', exist_ok=True)
        
        # Helper function to convert numpy/pytorch types to Python native types
        def convert_to_python_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            else:
                return obj
        
        results_data = {
            'rewards': experiment_results['rewards'],
            'fl_accuracies': experiment_results['fl_accuracies'],
            'fl_losses': experiment_results['fl_losses'],
            'training_times': experiment_results['training_times'],
            'assignments': experiment_results['assignments'],
            'epsilon_values': experiment_results['epsilon_values'],
            'ddqn_decisions': experiment_results['ddqn_decisions'],
            'final_decisions': experiment_results['final_decisions'],
            'num_rounds': ROUNDS,
            'num_edges': NUM_EDGES,
            'num_devices': NUM_DEVICES,
            'warmup_rounds': WARMUP_ROUNDS,
            'final_epsilon': epsilon,
            'lambda_weights': LAMBDA
        }
        
        # Convert all numpy/pytorch types to Python native types
        results_data = convert_to_python_types(results_data)
        
        with open('results/experiment_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print_success("Results saved to results/experiment_results.json")

    def save_incremental_results(self, round_num):
        """Save results incrementally after each round"""
        os.makedirs('results', exist_ok=True)
        
        # Helper function to convert numpy/pytorch types to Python native types
        def convert_to_python_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            else:
                return obj
        
        incremental_data = {
            'rewards': experiment_results['rewards'],
            'fl_accuracies': experiment_results['fl_accuracies'],
            'fl_losses': experiment_results['fl_losses'],
            'training_times': experiment_results['training_times'],
            'assignments': experiment_results['assignments'],
            'epsilon_values': experiment_results['epsilon_values'],
            'ddqn_decisions': experiment_results['ddqn_decisions'],
            'ilp_decisions': experiment_results['ilp_decisions'],
            'final_decisions': experiment_results['final_decisions'][:round_num] if round_num <= len(experiment_results['final_decisions']) else experiment_results['final_decisions'],
            'current_round': round_num,
            'total_rounds': ROUNDS,
            'num_edges': NUM_EDGES,
            'num_devices': NUM_DEVICES,
            'current_epsilon': epsilon,
            'lambda_weights': LAMBDA,
            'last_saved': datetime.now().isoformat()
        }
        
        # Convert all numpy/pytorch types to Python native types
        incremental_data = convert_to_python_types(incremental_data)
        
        # Save incremental file
        with open('results/incremental_results.json', 'w') as f:
            json.dump(incremental_data, f, indent=2)
        
        # Also save round-specific file
        with open(f'results/round_{round_num:03d}_results.json', 'w') as f:
            json.dump(incremental_data, f, indent=2)
        
        print_success(f"Incremental results saved after round {round_num}")

def serve():
    global cloud_service
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    cloud_service = CloudCoreService()
    federated_learning_pb2_grpc.add_CloudCoreServicer_to_server(cloud_service, server)
   
    server.add_insecure_port(f'{CLOUD_IP}:{CLOUD_PORT}')
    server.start()
   
    print_header("🚀 CLOUD CORE STARTED")
    print_metric("Internal Address", f"{CLOUD_IP}:{CLOUD_PORT}")
    print_metric("External Address", f"{EXTERNAL_IP}:{CLOUD_PORT}")
    print_metric("Hybrid Algorithm", f"DDQN+ILP (Warmup={WARMUP_ROUNDS} rounds)")
    print_metric("Mathematical Model", "Following DDQN & ILP equations")
    print_metric("Lambda Weights", f"λ₁={LAMBDA[0]}, λ₂={LAMBDA[1]}, λ₃={LAMBDA[2]}")
   
    time.sleep(3)
   
    def run_experiment():
        successful_rounds = 0
       
        for round_num in range(1, ROUNDS + 1):
            try:
                success = cloud_service.run_fl_round(round_num)
                if success:
                    successful_rounds += 1

                cloud_service.save_incremental_results(round_num)

                if round_num < ROUNDS:
                    print_info(f"Next round in 30 seconds...")
                    time.sleep(30)
                   
            except Exception as e:
                print_error(f"Error in round {round_num}: {e}")
                import traceback
                traceback.print_exc()
                continue
       
        print_header("📊 EXPERIMENT COMPLETE")
        print_metric("Successful rounds", f"{successful_rounds}/{ROUNDS}")
       
        if experiment_results['rewards']:
            avg_reward = np.mean(experiment_results['rewards'])
            avg_accuracy = np.mean(experiment_results['fl_accuracies'])
            avg_loss = np.mean(experiment_results['fl_losses'])
           
            print_subheader("FINAL STATISTICS")
            print_metric("Average Reward", f"{avg_reward:.3f}")
            print_metric("Average FL Accuracy", f"{avg_accuracy:.4f}")
            print_metric("Average FL Loss", f"{avg_loss:.4f}")
            print_metric("Final Epsilon", f"{epsilon:.3f}")
       
        # Analyze decisions
        print_subheader("DECISION ANALYSIS")
        ilp_used = sum(1 for d in experiment_results['final_decisions'] if 'ILP' in d.get('decision_reason', ''))
        ddqn_used = sum(1 for d in experiment_results['final_decisions'] if 'DDQN' in d.get('decision_reason', ''))
        consensus = sum(1 for d in experiment_results['final_decisions'] if 'Consensus' in d.get('decision_reason', ''))
       
        print_metric("ILP Decisions", f"{ilp_used}")
        print_metric("DDQN Decisions", f"{ddqn_used}")
        print_metric("Consensus Decisions", f"{consensus}")
       
        cloud_service.save_experiment_results()
        print_success("🎉 Experiment completed successfully!")
   
    experiment_thread = threading.Thread(target=run_experiment, daemon=True)
    experiment_thread.start()
   
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n👋 Cloud Core shutting down...")
        server.stop(0)


if __name__ == '__main__':
    serve()



