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
* `--no-progress`: Disable the terminal progress indicator while fetching abstracts/arXiv data.
* `--output`: File to write the JSON output to (default is standard output).

All network requests automatically retry up to three times to mitigate transient connectivity issues.

Example:

```bash
python nips_scraper.py --year 2023 --limit 10 --output papers.json
```

The script outputs JSON records with paper metadata including title, authors, track, abstract URL, PDF URL, abstract text (unless `--skip-abstracts` is used), and when found, an `arxiv` object containing identifier, URL, PDF link, authors, summary, publication dates, and category information. The scraper now performs multiple arXiv searches with increasingly permissive title token queries so that matches are found even when punctuation or accenting differs between the NeurIPS site and arXiv. Even if a paper's abstract page temporarily fails to load, the scraper will continue querying the arXiv API so metadata is still attached whenever a match can be located. When run in an interactive terminal a progress indicator is displayed while optional metadata is retrieved. Use `--no-progress` to suppress it if desired.
