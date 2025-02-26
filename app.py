import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import pycountry
from Bio import Entrez
import time
import re
import logging
from typing import Dict, List, Set, Tuple, Optional
import os
from difflib import get_close_matches
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import json
import datetime
from pathlib import Path

st.set_page_config(page_title="PubMed Research Trends", layout="wide")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

Entrez.email = "your_email@example.com"  # Replace with your email
SEARCH_TERMS = ["machine learning", "myosin", "transient ischemic attack", "extracellular vesicles"]
MAX_RESULTS = 25
DEFAULT_IMPACT_FACTOR = 1.0
PREPRINT_IMPACT_FACTOR = 0.5
RATE_LIMIT_DELAY = 0.5
FUZZY_MATCH_CUTOFF = 0.80

JOURNAL_PREFIXES = {'the ', 'a ', 'an '}

PREPRINT_SERVERS = {
    'arxiv': 'arxiv',
    'biorxiv': 'biorxiv',
    'medrxiv': 'medrxiv',
    'chemrxiv': 'chemrxiv',
    'researchsquare': 'research square',
    'ssrn': 'ssrn',
    'preprint': 'preprint server'
}

JOURNAL_ABBREVIATIONS = {
    'intl': 'international', 'int': 'international', 'am': 'american',
    'j': 'journal', 'rev': 'review', 'res': 'research',
    'sci': 'science', 'med': 'medical', 'natl': 'national',
    'biol': 'biology', 'biochem': 'biochemistry', 'chem': 'chemistry',
    'phys': 'physical', 'acad': 'academy', 'proc': 'proceedings',
    'lancet': 'the lancet', 'ann': 'annals', 'appl': 'applied',
    'clin': 'clinical', 'assoc': 'association', 'biomed': 'biomedical',
    'dev': 'development', 'exp': 'experimental', 'genet': 'genetics',
    'immunol': 'immunology', 'mol': 'molecular', 'neurosci': 'neuroscience',
    'pharmacol': 'pharmacology', 'physiol': 'physiology', 'psychol': 'psychology',
    'ther': 'therapy', 'toxicol': 'toxicology', 'arch': 'archives',
    'tech': 'technology', 'cardiol': 'cardiology', 'urol': 'urology'
}

COMMON_JOURNALS = {
    'lancet': 'the lancet',
    'bmj': 'british medical journal',
    'jama': 'journal of the american medical association',
    'nejm': 'new england journal of medicine',
    'nature': 'nature',
    'science': 'science',
    'pnas': 'proceedings of the national academy of sciences',
    'cell': 'cell',
    'angewandte chemie': 'angewandte chemie international edition',
    'nejm': 'new england journal of medicine'
}

MANUAL_MAPPINGS = {
    'angewandte chemie (international ed. in english)': 'angewandte chemie international edition',
    'biochemistry. biokhimiia': 'biochemistry',
    'archives of medical sciences. atherosclerotic diseases': 'archives of medical science',
    'arthroscopy : the journal of arthroscopic & related surgery': 'arthroscopy',
    'bioelectronic medicine': 'bioelectronic medicine',
    'biorxiv : the preprint server for biology': 'biorxiv',
    'canadian urological association journal': 'canadian urological association journal',
    'advances in drug and alcohol research': 'drug and alcohol research',
    'cen case reports': 'cen case reports'
}

COUNTRY_VARIATIONS = {
    'P.R. China': 'China', 'PR China': 'China', 'Peoples Republic of China': 'China',
    "People's Republic of China": 'China', 'mainland china': 'China',
    'republic of china': 'China', 'taiwan': 'Taiwan', 'hong kong': 'Hong Kong',
    'usa': 'United States', 'united states': 'United States', 'u.s.a.': 'United States',
    'uk': 'United Kingdom', 'u.k.': 'United Kingdom', 'england': 'United Kingdom',
    'korea': 'South Korea', 'republic of korea': 'South Korea'
}

COMPILED_COUNTRY_PATTERNS = {
    re.compile(rf'\b{re.escape(k)}\b', re.IGNORECASE): v 
    for k, v in COUNTRY_VARIATIONS.items()
}

def clean_journal_name(name: str) -> str:
    for src, target in MANUAL_MAPPINGS.items():
        if src.lower() in name.lower():
            return target
            
    for preprint, replace in PREPRINT_SERVERS.items():
        if preprint.lower() in name.lower():
            return replace
    
    name = name.split('=')[0]
    name = re.sub(r'\([^)]*\)', '', name)
    
    for common, full in COMMON_JOURNALS.items():
        if common.lower() in name.lower():
            return full
    
    name = re.sub(r'\s*:\s*.*$', '', name)
    name = re.sub(r'\s*=\s*.*$', '', name)
    
    parts = name.split(',')
    if len(parts) > 1:
        cleaned_parts = [parts[0]]
        location_hints = {'england', 'london', 'uk', 'u.s.a', 'usa', 'germany', 'switzerland', 
                         'netherlands', 'china', 'japan', 'france', 'italy', 'canada', 'australia'}
        for part in parts[1:]:
            part_lower = part.lower().strip()
            if not any(hint in part_lower for hint in location_hints):
                if not any(month in part_lower for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']):
                    cleaned_parts.append(part)
        name = ','.join(cleaned_parts)
    
    name = re.sub(r'\s*:\s*(?:the\s+)?official\s+(?:journal|publication)\s+of\s+.*$', '', name, flags=re.IGNORECASE)
    name = ' '.join(word for word in name.split())
    
    lower_name = name.lower()
    for prefix in JOURNAL_PREFIXES:
        if lower_name.startswith(prefix):
            return name
    
    if any(journal in lower_name for journal in ['lancet', 'bmj']) and not lower_name.startswith('the '):
        name = 'The ' + name
        
    name = re.sub(r'^the journal of (.+) journal$', r'the \1 journal', name, flags=re.IGNORECASE)
    
    return name.strip()

def normalize_journal_name(name: str) -> str:
    name = clean_journal_name(name)
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    words = name.split()
    normalized = []
    for word in words:
        normalized.append(JOURNAL_ABBREVIATIONS.get(word, word))
    return ' '.join(normalized)

def fuzzy_match_journal(journal: str, impact_factors: Dict) -> Optional[str]:
    normalized_name = normalize_journal_name(journal)
    
    if normalized_name.lower() in PREPRINT_SERVERS.values():
        return None
    
    normalized_keys = {normalize_journal_name(k): k for k in impact_factors.keys()}
    
    if normalized_name in normalized_keys:
        return normalized_keys[normalized_name]
    
    for key in normalized_keys:
        if normalized_name in key or key in normalized_name:
            if len(normalized_name) >= 5 and len(key) >= 5:
                return normalized_keys[key]
    
    matches = get_close_matches(normalized_name, normalized_keys.keys(), n=1, cutoff=FUZZY_MATCH_CUTOFF)
    if matches:
        return normalized_keys[matches[0]]
    
    return None

def load_impact_factors(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, sep=";", encoding="utf-8")
        else:
            df = pd.read_csv(st.session_state.get('sjr_path', "scimagojr_2023.csv"), sep=";", encoding="utf-8")
        
        df.rename(columns=lambda x: x.strip(), inplace=True)
        if not {"Title", "SJR", "Country"}.issubset(df.columns):
            raise ValueError("Expected columns 'Title', 'SJR', and 'Country' not found in CSV.")
        df["SJR"] = df["SJR"].str.replace(",", ".") if df["SJR"].dtype == "object" else df["SJR"]
        df["SJR"] = pd.to_numeric(df["SJR"], errors="coerce")
        df["SJR"] = df["SJR"].fillna(DEFAULT_IMPACT_FACTOR)
        df["Title"] = df["Title"].str.lower().str.strip()
        df["Country"] = df["Country"].str.strip()
        df = df.groupby("Title", as_index=False).agg({"SJR": "max", "Country": "first"})
        impact_factors = df.set_index("Title")[["SJR", "Country"]].to_dict(orient="index")
        return impact_factors
    except Exception as e:
        st.error(f"Error loading impact factors: {e}")
        return {}

def extract_countries_from_affiliation(affiliation: str) -> Set[str]:
    if not affiliation:
        return set()

    countries = set()
    parts = [p.strip() for p in affiliation.replace(';', ',').split(',')]
    
    for part in parts:
        for pattern, country in COMPILED_COUNTRY_PATTERNS.items():
            if pattern.search(part):
                countries.add(country)
                break
        
        if not countries:
            try:
                country_obj = pycountry.countries.lookup(part)
                countries.add(country_obj.name)
            except LookupError:
                try:
                    subdivision = pycountry.subdivisions.lookup(part)
                    countries.add(subdivision.country.name)
                except LookupError:
                    pass
    
    return countries

def query_pubmed(term: str, max_results: int = MAX_RESULTS) -> List[Tuple[str, str, str]]:
    papers = []
    try:
        with Entrez.esearch(db="pubmed", term=term, retmax=max_results) as handle:
            record = Entrez.read(handle)
            pubmed_ids = record["IdList"]

        for pubmed_id in pubmed_ids:
            try:
                with Entrez.efetch(db="pubmed", id=pubmed_id, rettype="medline", retmode="text") as handle:
                    paper_data = handle.read()
                
                journal = "Unknown"
                title = ""
                if "JT  - " in paper_data:
                    journal = paper_data.split("JT  - ")[1].split("\n")[0]
                elif "TA  - " in paper_data:
                    journal = paper_data.split("TA  - ")[1].split("\n")[0]
                if "TI  - " in paper_data:
                    title = paper_data.split("TI  - ")[1].split("\n")[0]
                
                is_preprint = any(server in journal.lower() for server in PREPRINT_SERVERS) or \
                             any(server in title.lower() for server in PREPRINT_SERVERS)
                
                affiliations = []
                affiliation_fields = ["AD  - ", "AUID- ", "FAU - ", "AU  - ", "PL  - "]
                for field in affiliation_fields:
                    if field in paper_data:
                        blocks = paper_data.split(field)[1:]
                        field_affiliations = [block.split("\n")[0] for block in blocks]
                        affiliations.extend(field_affiliations)

                countries = set()
                for aff in affiliations:
                    extracted = extract_countries_from_affiliation(aff)
                    if extracted:
                        countries.update(extracted)
                
                if not countries:
                    countries.add("Unknown")
                
                for country in countries:
                    papers.append((journal.strip().lower(), country, term))
                
                time.sleep(RATE_LIMIT_DELAY)
                
            except Exception as e:
                continue

    except Exception as e:
        st.error(f"Error in PubMed search for term '{term}': {str(e)}")

    return papers

def fetch_all_pubmed_data(search_terms, progress_bar=None) -> List[Tuple[str, str, str]]:
    all_data = []
    for i, term in enumerate(search_terms):
        if progress_bar:
            progress_bar.progress((i) / len(search_terms), text=f"Searching for '{term}'...")
        papers = query_pubmed(term)
        all_data.extend(papers)
        if progress_bar:
            progress_bar.progress((i + 1) / len(search_terms))
    return all_data

def assign_impact_factors(pubmed_data: List[Tuple[str, str, str]], 
                         impact_factors: Dict) -> Tuple[List[Dict], Dict[str, float], List[Dict]]:
    processed_data = []
    missing_journals = {}
    matched_journals = {}
    preprint_data = []
    
    for journal, country, subject in pubmed_data:
        original_name = journal.lower()
        is_preprint = any(server in original_name for server in PREPRINT_SERVERS.values())
        
        if is_preprint:
            impact_factor = PREPRINT_IMPACT_FACTOR
            data_entry = {
                "Journal": journal,
                "Country": country,
                "Subject": subject,
                "Impact Factor": impact_factor
            }
            processed_data.append(data_entry)
            preprint_data.append(data_entry)
            continue
            
        matched_name = fuzzy_match_journal(journal, impact_factors)
        if matched_name:
            impact_factor = impact_factors[matched_name]["SJR"]
            matched_journals[original_name] = matched_name
        else:
            impact_factor = DEFAULT_IMPACT_FACTOR
            missing_journals[original_name] = impact_factor
        
        data_entry = {
            "Journal": journal,
            "Country": country,
            "Subject": subject,
            "Impact Factor": impact_factor
        }
        processed_data.append(data_entry)
        
        if is_preprint:
            preprint_data.append(data_entry)
            
    return processed_data, missing_journals, preprint_data

def save_data(processed_data, preprint_data, missing_journals):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    with open(data_dir / f"processed_data_{today}.json", "w") as f:
        json.dump(processed_data, f)
        
    with open(data_dir / f"preprint_data_{today}.json", "w") as f:
        json.dump(preprint_data, f)
        
    with open(data_dir / f"missing_journals_{today}.json", "w") as f:
        json.dump(missing_journals, f)
    
    with open(data_dir / "latest_run.txt", "w") as f:
        f.write(today)
        
    return today

def load_saved_data():
    data_dir = Path("data")
    
    if not data_dir.exists():
        return None, None, None, None
    
    latest_file = data_dir / "latest_run.txt"
    if not latest_file.exists():
        return None, None, None, None
    
    with open(latest_file, "r") as f:
        latest_date = f.read().strip()
    
    processed_file = data_dir / f"processed_data_{latest_date}.json"
    preprint_file = data_dir / f"preprint_data_{latest_date}.json"
    missing_file = data_dir / f"missing_journals_{latest_date}.json"
    
    if not all(f.exists() for f in [processed_file, preprint_file, missing_file]):
        return None, None, None, None
    
    with open(processed_file, "r") as f:
        processed_data = json.load(f)
        
    with open(preprint_file, "r") as f:
        preprint_data = json.load(f)
        
    with open(missing_file, "r") as f:
        missing_journals = json.load(f)
    
    return processed_data, preprint_data, missing_journals, latest_date

def create_visualizations(processed_data, missing_journals, preprint_data):
    if not processed_data:
        st.warning("No data to visualize")
        return
    
    df = pd.DataFrame(processed_data)
    
    preprint_df = pd.DataFrame(preprint_data) if preprint_data else pd.DataFrame()
    
    # Plot general choropleth maps
    raw_counts = df.groupby(["Country", "Subject"]).size().reset_index(name="Count")
    if_weighted = df.groupby(["Country", "Subject"])["Impact Factor"].sum().reset_index()

    st.subheader("Publication Counts by Country & Research Area")
    fig_raw = px.choropleth(
        raw_counts,
        locations="Country",
        locationmode="country names",
        color="Count",
        facet_col="Subject",
        hover_name="Country",
        title="Number of Articles by Country & Subject (Raw Counts)",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_raw, use_container_width=True)

    st.subheader("Impact Factor-Weighted Publications by Country & Research Area")
    fig_if = px.choropleth(
        if_weighted,
        locations="Country",
        locationmode="country names",
        color="Impact Factor",
        facet_col="Subject",
        hover_name="Country",
        title="Impact Factor-Weighted Articles by Country & Subject",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_if, use_container_width=True)

    # Plot preprint-specific data
    if not preprint_df.empty:
        preprint_counts = preprint_df.groupby(["Country", "Subject"]).size().reset_index(name="Count")
        
        valid_countries = preprint_counts["Country"].nunique() > 1 or preprint_counts["Country"].iloc[0] != "Unknown"
        if valid_countries:
            st.subheader("Preprint Distribution by Country & Research Area")
            fig_preprint = px.choropleth(
                preprint_counts,
                locations="Country",
                locationmode="country names",
                color="Count",
                facet_col="Subject",
                hover_name="Country",
                title="Preprint Distribution by Country & Subject",
                color_continuous_scale="Plasma",
            )
            st.plotly_chart(fig_preprint, use_container_width=True)

        # First identify all preprints
        preprint_journals = set(preprint_df["Journal"].unique())
        
        # Create comparison data using original processed data
        comparison_data = []
        for entry in processed_data:
            is_preprint = entry["Journal"].lower() in preprint_journals
            comparison_data.append({
                "Country": entry["Country"],
                "Subject": entry["Subject"],
                "Type": "Preprint" if is_preprint else "Peer-Reviewed",
                "Count": 1
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_counts = comparison_df.groupby(["Country", "Subject", "Type"]).size().reset_index(name="Count")
        
        # Only plot if we have meaningful comparison data
        if len(comparison_counts) > 0:
            st.subheader("Preprint vs. Peer-Reviewed Articles Comparison")
            fig_comparison = px.bar(
                comparison_counts,
                x="Country",
                y="Count",
                color="Type",
                facet_col="Subject",
                title="Preprint vs. Peer-Reviewed Articles by Country & Subject",
                barmode="group"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

    # Plot missing journals
    if missing_journals:
        st.subheader("Unmatched Journals Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Word Cloud of Unmatched Journals")
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100
            ).generate_from_frequencies(Counter(missing_journals))
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
        with col2:
            st.write("Top 20 Unmatched Journals")
            top_missing = dict(sorted(Counter(missing_journals).items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:20])
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(len(top_missing)), list(top_missing.values()))
            ax.set_xticks(range(len(top_missing)))
            ax.set_xticklabels(list(top_missing.keys()), rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

def main():
    st.title("PubMed Research Trends Analysis")
    
    st.sidebar.title("Configuration")
    
    # Check if data already exists
    saved_data = load_saved_data()
    if saved_data[0] is not None:
        processed_data, preprint_data, missing_journals, latest_date = saved_data
        st.sidebar.success(f"Using saved data from {latest_date}")
        run_new_analysis = st.sidebar.checkbox("Run new analysis?", False)
    else:
        run_new_analysis = True
    
    # File upload for SJR CSV
    uploaded_file = st.sidebar.file_uploader("Upload SJR CSV file (optional)", type=["csv"])
    if uploaded_file is not None:
        st.session_state['sjr_path'] = uploaded_file
        st.sidebar.success("SJR CSV file uploaded successfully!")
    else:
        st.sidebar.text("Using default SJR CSV file")
        if 'sjr_path' not in st.session_state:
            st.session_state['sjr_path'] = "scimagojr_2023.csv"
    
    # Custom search terms
    default_terms = ", ".join(SEARCH_TERMS)
    search_terms_input = st.sidebar.text_area("Search Terms (comma-separated)", value=default_terms)
    search_terms = [term.strip() for term in search_terms_input.split(",")]
    
    # Max results per search term
    max_results = st.sidebar.number_input("Max results per search term", min_value=5, max_value=100, value=MAX_RESULTS)
    
    if run_new_analysis:
        if st.sidebar.button("Run Analysis"):
            impact_factors = load_impact_factors(uploaded_file)
            if not impact_factors:
                st.error("Failed to load impact factors. Please check the CSV file.")
                return
                
            progress_bar = st.progress(0, text="Fetching PubMed data...")
            pubmed_data = fetch_all_pubmed_data(search_terms, progress_bar)
            
            if not pubmed_data:
                st.error("No data retrieved from PubMed. Please check your search terms.")
                return
                
            st.info(f"Processing {len(pubmed_data)} articles...")
            processed_data, missing_journals, preprint_data = assign_impact_factors(pubmed_data, impact_factors)
            
            save_date = save_data(processed_data, preprint_data, missing_journals)
            st.success(f"Data saved successfully! Date: {save_date}")
            
            create_visualizations(processed_data, missing_journals, preprint_data)
        else:
            if saved_data[0] is not None:
                create_visualizations(saved_data[0], saved_data[2], saved_data[1])
            else:
                st.info("Upload SJR CSV file and click 'Run Analysis' to start.")
    else:
        create_visualizations(processed_data, missing_journals, preprint_data)
    
if __name__ == "__main__":
    main()