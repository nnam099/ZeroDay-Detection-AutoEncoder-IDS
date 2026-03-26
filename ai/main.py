import yaml
import torch
from models.transformer_vae import TransformerVAE
from scripts.monitor_deamon import start_monitoring

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Khởi tạo Model
    model = TransformerVAE(
        feature_size=config['model']['feature_size'],
        sequence_length=config['model']['seq_length'],
        latent_dim=config['model']['latent_dim']
    ).cuda()

    # Load trọng số đã huấn luyện
    model.load_state_dict(torch.load(config['model']['model_path']))
    model.eval()

    # Optimizer phục vụ cho Self-learning
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['lr'])

    # Bắt đầu giám sát
    start_monitoring(model, optimizer, config)

if __name__ == "__main__":
    main()