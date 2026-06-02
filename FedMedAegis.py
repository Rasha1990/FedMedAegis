import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import hashlib
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import json

# ===============================
# Core Data Structures
# ===============================
@dataclass
class ModelUpdate:
    client_id: str
    weights: Dict[str, torch.Tensor]
    timestamp: float
    round_number: int
    hash_value: str

class DAGNode:
    def __init__(self, model_update: ModelUpdate, parent_hashes: List[str]):
        self.model_update = model_update
        self.parent_hashes = parent_hashes
        self.timestamp = time.time()
        self.node_hash = self.compute_hash()
        self.validated = False

    def compute_hash(self) -> str:
        content = f"{self.model_update.hash_value} {self.parent_hashes} {self.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()

# ===============================
# Privacy Mechanisms (Differential Privacy)
# ===============================
class DifferentialPrivacy:
    def __init__(self, epsilon: float = 1.2, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta

    def add_noise(self, tensor: torch.Tensor, sensitivity: float = 1.0) -> torch.Tensor:
        if self.epsilon <= 0:
            return tensor
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        noise = torch.normal(0, sigma, tensor.shape)
        return tensor + noise

# ===============================
# Split Learning Components
# ===============================
class SplitNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, split_layer: int):
        super().__init__()
        self.split_layer = split_layer
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build client-side model
        client_layers = []
        dims = [input_dim] + hidden_dims[:split_layer]
        for i in range(len(dims) - 1):
            client_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        self.client_model = nn.Sequential(*client_layers)

        # Build server-side model
        server_layers = []
        if split_layer < len(hidden_dims):
            server_input_dim = hidden_dims[split_layer - 1]
            server_dims = hidden_dims[split_layer:] + [output_dim]
            dims = [server_input_dim] + server_dims
            for i in range(len(dims) - 1):
                if i == len(dims) - 2:
                    server_layers.append(nn.Linear(dims[i], dims[i+1]))
                else:
                    server_layers.extend([
                        nn.Linear(dims[i], dims[i+1]),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ])
        else:
            server_layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.server_model = nn.Sequential(*server_layers)

    def forward_client(self, x: torch.Tensor) -> torch.Tensor:
        return self.client_model(x)

    def forward_server(self, x: torch.Tensor) -> torch.Tensor:
        return self.server_model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        client_output = self.forward_client(x)
        server_output = self.forward_server(client_output)
        return server_output

# ===============================
# DAG-Based Blockchain
# ===============================
class DAGBlockchain:
    def __init__(self):
        self.dag = {}
        self.tips = set()
        self.validated_nodes = set()

    def add_node(self, model_update: ModelUpdate, parent_hashes: List[str] = None) -> Optional[str]:
        if parent_hashes is None:
            parent_hashes = list(self.tips) if self.tips else []
        
        node = DAGNode(model_update, parent_hashes)
        
        if self._validate_node(node):
            self.dag[node.node_hash] = node
            node.validated = True
            self.validated_nodes.add(node.node_hash)
            
            for parent_hash in parent_hashes:
                self.tips.discard(parent_hash)
            self.tips.add(node.node_hash)
            
            return node.node_hash
        
        return None

    def _validate_node(self, node: DAGNode) -> bool:
        for parent_hash in node.parent_hashes:
            if parent_hash not in self.validated_nodes and parent_hash != "":
                return False
        return True

# ===============================
# Main Secure FL Framework
# ===============================
class SecureFLFramework:
    def __init__(self, model_config: Dict, privacy_config: Dict):
        self.model = SplitNN(**model_config)
        self.dp = DifferentialPrivacy(**privacy_config.get('dp', {}))
        self.dag_blockchain = DAGBlockchain()
        self.current_round = 0
        self.global_weights = self._get_model_weights()

    def _get_model_weights(self) -> Dict[str, torch.Tensor]:
        return {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def set_model_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.copy_(weights[name])

    def train_round(self, available_clients: List[str], client_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Dict:
        print(f"Starting round {self.current_round + 1}")
        
        # Basic random client selection (Reputation system removed to protect main contribution)
        num_to_select = max(2, len(available_clients) // 2)
        selected_clients = random.sample(available_clients, num_to_select)

        client_updates = []
        for client_id in selected_clients:
            if client_id in client_data:
                update = self._client_training(client_id, client_data[client_id])
                if update:
                    client_updates.append(update)

        if not client_updates:
            return {"status": "failed", "reason": "no_updates"}

        # Basic Federated Averaging (MAML Aggregator removed to protect main contribution)
        aggregated_weights = {}
        for name in client_updates[0]['weights'].keys():
            aggregated_weights[name] = torch.stack([u['weights'][name] for u in client_updates]).mean(dim=0)
        
        self.set_model_weights(aggregated_weights)
        self.global_weights = aggregated_weights

        # Add to DAG Blockchain
        for update in client_updates:
            model_update = ModelUpdate(
                client_id=update['client_id'],
                weights=update['weights'],
                timestamp=time.time(),
                round_number=self.current_round,
                hash_value=update['hash']
            )
            self.dag_blockchain.add_node(model_update)

        self.current_round += 1
        
        return {
            "status": "success",
            "round": self.current_round,
            "participating_clients": len(client_updates)
        }

    def _client_training(self, client_id: str, data: Tuple[torch.Tensor, torch.Tensor]) -> Optional[Dict]:
        try:
            X, y = data
            
            local_model = SplitNN(
                input_dim=X.shape[1], 
                hidden_dims=self.model.hidden_dims, 
                output_dim=self.model.output_dim, 
                split_layer=self.model.split_layer
            )
            local_model.load_state_dict(self.model.state_dict())

            optimizer = optim.Adam(local_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            local_model.train()
            for epoch in range(3):
                client_output = local_model.forward_client(X)
                
                # Apply Differential Privacy
                private_output = self.dp.add_noise(client_output)
                server_output = local_model.forward_server(private_output)
                
                loss = criterion(server_output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Extract weights with privacy
            local_weights = {name: param.clone().detach() for name, param in local_model.named_parameters()}
            private_weights = {}
            for name, weight in local_weights.items():
                private_weights[name] = self.dp.add_noise(weight)

            # Compute hash for integrity
            weight_str = json.dumps(
                {k: v.cpu().numpy().tolist() for k, v in private_weights.items()}, 
                sort_keys=True
            )
            hash_value = hashlib.sha256(weight_str.encode()).hexdigest()

            return {
                'client_id': client_id,
                'weights': private_weights,
                'hash': hash_value
            }
            
        except Exception as e:
            print(f"Error in client {client_id}: {e}")
            return None

    def evaluate_global_model(self, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        X_test, y_test = test_data
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(X_test)
            pred_labels = torch.argmax(predictions, dim=1)
            accuracy = (pred_labels == y_test).float().mean().item()

            y_true = y_test.cpu().numpy()
            y_pred = pred_labels.cpu().numpy()
            y_prob = torch.softmax(predictions, dim=1).cpu().numpy()

            f1 = f1_score(y_true, y_pred, average='weighted')
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'mae': mae,
            'rmse': rmse,
            'current_round': self.current_round
        }

    def get_system_stats(self) -> Dict:
        return {
            'current_round': self.current_round,
            'blockchain_nodes': len(self.dag_blockchain.dag),
            'validated_nodes': len(self.dag_blockchain.validated_nodes),
            'privacy_epsilon': self.dp.epsilon,
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }

# ===============================
# Synthetic Data Generator
# ===============================
def create_synthetic_data(num_clients: int = 10, samples_per_client: int = 1000):
    client_data = {}
    for i in range(num_clients):
        if i < num_clients // 2:
            X = np.random.normal(0, 1, (samples_per_client, 20))
            threshold = 1
        else:
            X = np.random.normal(2, 1.5, (samples_per_client, 20))
            threshold = 2
        
        y = (X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, samples_per_client) > threshold).astype(int)
        client_data[f"client_{i}"] = (torch.FloatTensor(X), torch.LongTensor(y))
    
    return client_data

# ===============================
# Main Execution
# ===============================
def main():
    print("Initializing Secure FL Framework...")
    
    model_config = {
        'input_dim': 20,
        'hidden_dims': [128, 64, 32, 16],
        'output_dim': 2,
        'split_layer': 2
    }
    
    privacy_config = {
        'dp': {'epsilon': 1.2, 'delta': 1e-5}
    }

    framework = SecureFLFramework(model_config, privacy_config)
    client_data = create_synthetic_data(num_clients=10, samples_per_client=1000)

    X_test = torch.randn(200, 20)
    y_test = torch.randint(0, 2, (200,))
    test_data = (X_test, y_test)

    print("Starting federated training...")
    
    for round_num in range(5):
        print(f"\n--- Round {round_num + 1} ---")
        available_clients = list(client_data.keys())
        
        round_result = framework.train_round(available_clients, client_data)
        
        if round_result['status'] == 'success':
            eval_results = framework.evaluate_global_model(test_data)
            system_stats = framework.get_system_stats()
            
            print(f"Accuracy: {eval_results['accuracy']:.3f}, "
                  f"F1: {eval_results['f1_score']:.3f}, "
                  f"Blockchain Nodes: {system_stats['blockchain_nodes']}")

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    final_stats = framework.get_system_stats()
    print(f"Privacy Epsilon: {final_stats['privacy_epsilon']}")
    print(f"Total Blockchain Nodes: {final_stats['blockchain_nodes']}")

if __name__ == "__main__":
    main()