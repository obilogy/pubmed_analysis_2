# PubMed Research Trends Analysis

A Streamlit application that analyzes PubMed publications to track research trends across countries and subjects.

## Features
- Fetches and analyzes PubMed data for specified research terms
- Maps publication patterns by country
- Compares impact factors across regions
- Identifies preprint vs. peer-reviewed publication patterns
- Daily automated updates

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Place the SJR CSV file in the main directory
3. Run the Streamlit app: `streamlit run app.py`
4. For daily updates, run: `python scheduler.py`