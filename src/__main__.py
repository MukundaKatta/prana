"""CLI for prana."""
import sys, json, argparse
from .core import Prana

def main():
    parser = argparse.ArgumentParser(description="Prana — Wearable-Free Vital Sign Estimation from Smartphone Camera. Heart rate, BP, SpO2, respiratory rate from a selfie video.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Prana()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.process(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"prana v0.1.0 — Prana — Wearable-Free Vital Sign Estimation from Smartphone Camera. Heart rate, BP, SpO2, respiratory rate from a selfie video.")

if __name__ == "__main__":
    main()
