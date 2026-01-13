import sys
import os
sys.path.append(os.path.abspath("."))

from src.surface_extractor import extract_live_spx_surface

print("Live SPX surface test:")
print(extract_live_spx_surface("20260112"))   # put any valid SPX expiry here
