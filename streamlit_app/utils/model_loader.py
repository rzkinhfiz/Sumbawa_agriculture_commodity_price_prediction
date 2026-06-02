import sys
from pathlib import Path

import torch
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.base_model import BaseLSTMModel

MODEL_DIR = REPO_ROOT / 'models'
MODEL_BEST_DIR = MODEL_DIR / 'best'


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _normalize_name(commodity_name: str) -> str:
    return commodity_name.strip().replace(' ', '_')


def _find_checkpoint(commodity_name: str, mode: str) -> Path:
    normalized = _normalize_name(commodity_name)
    pattern = f'*{normalized}_{mode.lower()}*.pth'
    candidates = sorted(MODEL_BEST_DIR.glob(pattern))
    if not candidates:
        candidates = sorted(MODEL_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f'Tidak menemukan checkpoint untuk {commodity_name} mode {mode} di {MODEL_BEST_DIR} atau {MODEL_DIR}'
        )
    return candidates[-1]


@st.cache_resource
def load_model(
    commodity_name: str,
    mode: str,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
):
    checkpoint_path = _find_checkpoint(commodity_name, mode)
    device = get_device()
    model = BaseLSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
