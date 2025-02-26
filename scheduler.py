import schedule
import time
import subprocess
import logging
import datetime
import os
from pathlib import Path

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"pubmed_scheduler_{datetime.datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def run_pubmed_analysis():
    try:
        logging.info("Starting scheduled PubMed analysis...")
        
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Run the streamlit app in headless mode to update the data
        result = subprocess.run(
            ["python", "pubmed_analysis_headless.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logging.info("PubMed analysis completed successfully")
            logging.info(f"Output: {result.stdout}")
        else:
            logging.error(f"PubMed analysis failed with error code {result.returncode}")
            logging.error(f"Error: {result.stderr}")
            
    except Exception as e:
        logging.error(f"Error running PubMed analysis: {e}")

# Schedule the job to run daily at 2 AM
schedule.every().day.at("02:00").do(run_pubmed_analysis)

logging.info("Scheduler started. Will run PubMed analysis daily at 2 AM.")

# Run once immediately on startup
run_pubmed_analysis()

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute