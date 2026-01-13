import time
from run_live_engine import main_step, logger

print("Live engine running — press Ctrl+C to stop and save report.\n")

try:
    while True:
        main_step()
        time.sleep(60)   # one decision per minute
except KeyboardInterrupt:
    fname = logger.dump()
    print(f"\nSession report saved to {fname}")
