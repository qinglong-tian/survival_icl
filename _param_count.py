import torch, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tabicl._model.tabicl import TabICL
from tabicl.survival import DiscreteTimeSurvivalHead

model = TabICL(
    max_classes=0, num_quantiles=50, embed_dim=128,
    col_num_blocks=3, col_nhead=4, col_num_inds=128,
    col_affine=False, col_feature_group='same', col_feature_group_size=3,
    col_target_aware=True, col_ssmax='qassmax-mlp-elementwise',
    row_num_blocks=3, row_nhead=8, row_num_cls=4,
    row_rope_base=100000, row_rope_interleaved=False,
    icl_num_blocks=12, icl_nhead=4, icl_ssmax='qassmax-mlp-elementwise',
    ff_factor=2, dropout=0.0, activation='gelu',
    norm_first=True, bias_free_ln=False, recompute=False, survival=True,
)

icl_dim = 128 * 4  # embed_dim * row_num_cls
model.icl_predictor.survival = True
model.icl_predictor.y_encoder = torch.nn.Linear(2, icl_dim)
model.icl_predictor.decoder = DiscreteTimeSurvivalHead(d_model=icl_dim, num_bins=50)

total = sum(p.numel() for p in model.parameters())
col_params = sum(p.numel() for p in model.col_embedder.parameters())
row_params = sum(p.numel() for p in model.row_interactor.parameters())
icl_params = sum(p.numel() for p in model.icl_predictor.parameters())

print(f'Total:          {total:>10,}')
print(f'ColEmbedding:   {col_params:>10,}  (FROZEN in Stage 2)')
print(f'RowInteraction: {row_params:>10,}  (FROZEN in Stage 2)')
print(f'ICLPredictor:   {icl_params:>10,}  (TRAINABLE in Stage 2)')
print(f'')
print(f'FROZEN:   {col_params+row_params:>10,}  ({(col_params+row_params)/total*100:.1f}%)')
print(f'TRAINABLE:{icl_params:>10,}  ({icl_params/total*100:.1f}%)')

# Sub-components
ye = sum(p.numel() for p in model.icl_predictor.y_encoder.parameters())
tf = sum(p.numel() for p in model.icl_predictor.tf_icl.parameters())
dec = sum(p.numel() for p in model.icl_predictor.decoder.parameters())
ln = sum(p.numel() for p in model.icl_predictor.ln.parameters()) if model.icl_predictor.ln is not None else 0
print(f'  y_encoder:     {ye:>10,}')
print(f'  tf_icl:        {tf:>10,}')
print(f'  decoder:       {dec:>10,}')
print(f'  LayerNorm:     {ln:>10,}')
