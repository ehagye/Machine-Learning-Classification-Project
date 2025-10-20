import argparse

def main():
    ap = argparse.ArgumentParser(description="ML Classification Project Runner")
    ap.add_argument("--all", action="store_true", help="Run full pipeline (coming soon)")
    args = ap.parse_args()
    if args.all:
        print("Full pipeline will be implemented in the next step.")
    else:
        print("Use --all (coming soon) or future subcommands.")

if __name__ == "__main__":
    main()
