"""Advanced usage with batch processing and benchmarking."""
import time
from src.core import Prana

def main():
    instance = Prana(config={
        "timeout": 30,
        "retries": 3,
        "debug": True,
    })

    print("=== Advanced prana Example ===\n")

    # Batch processing
    inputs = [f"data_{i}" for i in range(10)]
    results = [instance.process(input=inp, index=i) for i, inp in enumerate(inputs)]
    success = sum(1 for r in results if r.get("ok"))
    print(f"Processed {len(results)} items ({success} succeeded)")

    # Benchmark
    start = time.time()
    for i in range(1000):
        instance.process(iteration=i)
    elapsed = (time.time() - start) * 1000
    print(f"\n1000 ops in {elapsed:.0f}ms ({elapsed/1000:.2f}ms/op)")

    # Stats
    stats = instance.get_stats()
    print(f"\nTotal ops: {stats['ops']}")
    print(f"Log size: {stats['log_size']}")

    # Cleanup
    instance.reset()
    print("Reset complete")

if __name__ == "__main__":
    main()
