import torch

from portfolio_generator import decode_portfolio_tensor, NUM_LEGS

# Dummy portfolio tensor
# Columns: [type_raw, strike_offset, weight]
portfolio_tensor = torch.tensor([
    [ 1.0, -0.2,  0.5],
    [-1.0,  0.1, -0.3],
    [ 1.0,  0.0,  0.8],
    [-1.0, -0.1, -0.2],
])

spot = 100.0

legs = decode_portfolio_tensor(
    portfolio_tensor=portfolio_tensor,
    spot=spot,
)

print("Decoded legs:")
for leg in legs:
    print(leg)
