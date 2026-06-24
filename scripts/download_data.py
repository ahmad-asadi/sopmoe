import argparse
from src.data.downloader import DataDownloader

def main():
    parser = argparse.ArgumentParser(description="Download raw market data for the project.")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--dir", type=str, default="./data/raw", help="Directory to save CSVs")
    
    args = parser.parse_args()
    
    downloader = DataDownloader(data_dir=args.dir)
    downloader.download_all(args.start, args.end)

if __name__ == "__main__":
    main()
