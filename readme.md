# NeurIPS Accepted Papers Scraper

This repository contains a simple Python script that scrapes the NeurIPS (formerly NIPS) proceedings site for accepted papers in a given year and enriches results with arXiv metadata when available.

## Requirements

Install the required dependencies (requests and BeautifulSoup) with pip:

```bash
pip install -r requirements.txt
```

or directly

```bash
pip install requests beautifulsoup4
```

## Usage

Run the script from the command line, specifying the year you are interested in:

```bash
python nips_scraper.py --year 2023
```

### Options

* `--limit`: Limit the number of papers to fetch (useful for testing).
* `--skip-abstracts`: Skip downloading abstracts to reduce the number of HTTP requests.
* `--skip-arxiv`: Skip attempting to match papers to arXiv entries.
* `--output`: File to write the JSON output to (default is standard output).

Example:

```bash
python nips_scraper.py --year 2023 --limit 10 --output papers.json
```

The script outputs JSON records with paper metadata including title, authors, track, abstract URL, PDF URL, abstract text (unless `--skip-abstracts` is used), and when found, an `arxiv` object containing identifier, URL, PDF link, authors, summary, publication dates, and category information.
