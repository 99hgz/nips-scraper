# Conference Accepted Papers Scraper

This repository contains a simple Python script that scrapes accepted paper listings for major conferences. It currently supports the NeurIPS (formerly NIPS) proceedings site, the ICLR OpenReview listings, and the ACL Anthology (ACL, NAACL, and EMNLP events), enriching results with arXiv metadata when available.

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

To scrape a different conference, pass the `--conference` flag (default is `neurips`):

```bash
python nips_scraper.py --conference acl --year 2023
python nips_scraper.py --conference naacl --year 2024
```

### Options

* `--limit`: Limit the number of papers to fetch (useful for testing).
* `--skip-abstracts`: Skip downloading abstracts to reduce the number of HTTP requests.
* `--skip-arxiv`: Skip attempting to match papers to arXiv entries.
* `--no-progress`: Disable the terminal progress indicator while fetching abstracts/arXiv data.
* `--output`: File to write the JSON output to (default is standard output).
* `--conference`: Which conference to scrape (`neurips`, `iclr`, `acl`, `naacl`, or `emnlp`).

All network requests automatically retry up to three times to mitigate transient connectivity issues.

Examples:

```bash
python nips_scraper.py --year 2023 --limit 10 --output papers.json

python nips_scraper.py --conference acl --year 2023 --limit 10 --output acl_papers.json

python nips_scraper.py --conference emnlp --year 2023 --limit 5 --skip-arxiv

python nips_scraper.py --conference iclr --year 2023 --limit 10 --skip-arxiv
```

The script outputs JSON records with paper metadata including title, authors, track, abstract URL, PDF URL, abstract text (unless `--skip-abstracts` is used), and when found, an `arxiv` object containing identifier, URL, PDF link, authors, summary, publication dates, and category information. The scraper performs multiple arXiv searches with increasingly permissive title token queries so that matches are found even when punctuation or accenting differs between the source site and arXiv. Even if a paper's abstract page temporarily fails to load, the scraper will continue querying the arXiv API so metadata is still attached whenever a match can be located. When run in an interactive terminal a progress indicator is displayed while optional metadata is retrieved. Use `--no-progress` to suppress it if desired.

For older ICLR editions (for example 2020), OpenReview does not populate the inline `venue` field, so the scraper first collects the per-paper decision notes and then looks up the corresponding submissions. This extra pass means fetching the full list of accepted papers can take roughly a minute, which is expected behaviour. For more recent ICLR years the Blind Submission invitation sometimes lags behind the public listings, so the scraper now falls back to the OpenReview search API when the traditional feed is empty, querying both the `venueid` and `venue` metadata to harvest the latest accepted posters, spotlights, and orals.
