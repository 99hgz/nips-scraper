"""Scrape major machine learning and NLP conference proceedings.

The scraper currently understands NeurIPS (https://papers.neurips.cc),
ICLR (https://openreview.net), and ACL Anthology events
(https://aclanthology.org) including ACL, NAACL, and EMNLP listings,
extracting paper metadata including the title, authors, track, abstract
URL, PDF URL, and optionally arXiv metadata when a matching entry can be
found.

Example
-------
python nips_scraper.py --conference iclr --year 2023 --limit 5 --output papers.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote, urljoin
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

NEURIPS_BASE_URL = "https://papers.neurips.cc"
NEURIPS_LISTING_PATH_TEMPLATE = "/paper_files/paper/{year}"
ACL_BASE_URL = "https://aclanthology.org"
ACL_EVENT_PATH_TEMPLATE = "/events/{event}-{year}/"
ACL_STYLE_CONFERENCES = {"acl", "naacl", "emnlp"}
OPENREVIEW_API_NOTES_URL = "https://api.openreview.net/notes"
OPENREVIEW_API_NOTES_SEARCH_URL = "https://api.openreview.net/notes/search"
OPENREVIEW_BASE_URL = "https://openreview.net"
ABSTRACT_PATH_PATTERN = re.compile(
    r"/paper_files/paper/(?P<year>\d{4})/hash/(?P<hash>[0-9a-f]+)-Abstract-(?P<section>[^.]+)\.html"
)
ARXIV_ID_PATTERN = re.compile(r"arxiv\.org/(?:abs|pdf)/(?P<identifier>[^/?#]+)", re.IGNORECASE)
ARXIV_API_URL = "http://export.arxiv.org/api/query"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0


@dataclass
class ArxivMetadata:
    """Metadata returned by the arXiv API for a paper."""

    identifier: str
    url: str
    title: str
    authors: List[str]
    summary: str
    pdf_url: Optional[str]
    published: Optional[str]
    updated: Optional[str]
    primary_category: Optional[str]
    comment: Optional[str]


@dataclass
class Paper:
    """Container for NeurIPS paper metadata."""

    year: int
    title: str
    authors: List[str]
    track: str
    paper_url: str
    pdf_url: str
    abstract: Optional[str] = None
    arxiv: Optional[ArxivMetadata] = None


def _request_with_retries(
    url: str,
    params: Optional[dict[str, str]] = None,
    *,
    method: str = "GET",
    json_body: Optional[dict[str, Any]] = None,
) -> requests.Response:
    """Perform a GET request with retry logic for transient failures."""

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/html;q=0.9,*/*;q=0.8",
    }
    last_exception: Optional[requests.RequestException] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 429:
                if attempt >= MAX_RETRIES:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else RETRY_DELAY_SECONDS
                except ValueError:
                    delay = RETRY_DELAY_SECONDS
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response
        except requests.HTTPError:
            # HTTP status errors are unlikely to be transient; re-raise immediately.
            raise
        except requests.RequestException as exc:  # pragma: no cover - network issue
            last_exception = exc
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY_SECONDS)

    assert last_exception is not None  # pragma: no cover - defensive guard
    raise last_exception


def _get_soup(url: str) -> BeautifulSoup:
    """Fetch *url* and return a BeautifulSoup instance."""

    response = _request_with_retries(url)
    return BeautifulSoup(response.text, "html.parser")


def _normalise_track(section: str, li_classes: Iterable[str] | None) -> str:
    """Convert a section identifier to a human readable track label."""

    if section:
        return section.replace("_", " ")
    if li_classes:
        for cls in li_classes:
            if cls in {"conference", "datasets_and_benchmarks"}:
                return cls.replace("_", " ")
    return "Conference"


def _extract_authors(container: BeautifulSoup) -> List[str]:
    """Extract a list of authors from a list item."""

    author_tag = container.find("i")
    if not author_tag:
        return []
    authors = [name.strip() for name in author_tag.get_text(" ", strip=True).split(",")]
    return [name for name in authors if name]


def _extract_abstract_from_soup(soup: BeautifulSoup) -> Optional[str]:
    """Retrieve the abstract text from a parsed abstract page."""

    card = soup.find("div", class_=lambda value: isinstance(value, str) and "acl-abstract" in value)
    if card:
        text = card.get_text(" ", strip=True)
        if text:
            return text

    header = soup.find(
        lambda tag: tag.name in {"h2", "h3", "h4", "h5", "h6"}
        and "abstract" in tag.get_text(strip=True).lower()
    )
    if not header:
        return None

    paragraph = header.find_next("p")
    if not paragraph:
        return None

    text = paragraph.get_text(" ", strip=True)
    return text if text else None


def _extract_arxiv_identifier(soup: BeautifulSoup) -> Optional[str]:
    """Extract an arXiv identifier from an abstract page if available."""

    meta = soup.find("meta", attrs={"name": "citation_arxiv_id"})
    if meta and meta.get("content"):
        identifier = meta["content"].strip()
        if identifier:
            return identifier

    for anchor in soup.select('a[href*="arxiv.org/"]'):
        href = anchor.get("href", "")
        match = ARXIV_ID_PATTERN.search(href)
        if match:
            identifier = match.group("identifier")
            if identifier.lower().endswith(".pdf"):
                identifier = identifier[:-4]
            if identifier:
                return identifier

    text_match = soup.find(string=re.compile(r"arXiv:\s*(?P<identifier>[\w./-]+)", re.IGNORECASE))
    if text_match:
        match = re.search(r"arXiv:\s*(?P<identifier>[\w./-]+)", text_match, flags=re.IGNORECASE)
        if match:
            identifier = match.group("identifier").rstrip(".,")
            if identifier:
                return identifier

    return None


def _normalise_title(title: str) -> str:
    """Normalise a title for fuzzy matching."""

    cleaned = re.sub(r"[\W_]+", " ", title, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _title_similarity(lhs: str, rhs: str) -> float:
    """Return a similarity score between two titles in the range [0, 1]."""

    if not lhs or not rhs:
        return 0.0
    return SequenceMatcher(None, _normalise_title(lhs), _normalise_title(rhs)).ratio()


def _parse_arxiv_entry(entry: ET.Element) -> ArxivMetadata:
    """Convert an arXiv API entry into :class:`ArxivMetadata`."""

    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    def _find_text(path: str, default: Optional[str] = None) -> Optional[str]:
        element = entry.find(path, namespaces)
        if element is None or element.text is None:
            return default
        return element.text.strip() or default

    identifier = _find_text("atom:id", "") or ""
    identifier = identifier.rsplit("/", 1)[-1]

    title = _find_text("atom:title", "") or ""
    summary = _find_text("atom:summary", "") or ""
    published = _find_text("atom:published")
    updated = _find_text("atom:updated")
    comment = _find_text("arxiv:comment")

    authors = [
        author.text.strip()
        for author in entry.findall("atom:author/atom:name", namespaces)
        if author.text
    ]

    pdf_link = entry.find("atom:link[@title='pdf']", namespaces)
    pdf_url = pdf_link.get("href") if pdf_link is not None else None

    alternate_link = entry.find("atom:link[@rel='alternate']", namespaces)
    url = alternate_link.get("href") if alternate_link is not None else f"https://arxiv.org/abs/{identifier}"

    primary_category_tag = entry.find("arxiv:primary_category", namespaces)
    primary_category = primary_category_tag.get("term") if primary_category_tag is not None else None

    return ArxivMetadata(
        identifier=identifier,
        url=url,
        title=title.strip(),
        authors=authors,
        summary=summary.strip(),
        pdf_url=pdf_url,
        published=published,
        updated=updated,
        primary_category=primary_category,
        comment=comment,
    )


def _fetch_arxiv_metadata_by_id(identifier: str) -> Optional[ArxivMetadata]:
    """Fetch arXiv metadata for a specific identifier."""

    url = f"{ARXIV_API_URL}?id_list={quote(identifier)}"
    response = _request_with_retries(url)

    root = ET.fromstring(response.text)
    entry = root.find("{http://www.w3.org/2005/Atom}entry")
    if entry is None:
        return None
    return _parse_arxiv_entry(entry)


def _tokenise_title_for_search(title: str) -> List[str]:
    """Return a list of tokens suitable for constructing arXiv queries."""

    cleaned = re.sub(r"[^0-9A-Za-z]+", " ", title)
    tokens = [token for token in cleaned.split() if token]
    # arXiv performs better with a small number of filters; keep the first few
    return tokens[:8]


def _iter_arxiv_search_entries(queries: Iterable[str]) -> Iterable[ET.Element]:
    """Yield entries returned by executing *queries* against the arXiv API."""

    seen_ids: set[str] = set()
    for query in queries:
        if not query:
            continue
        url = f"{ARXIV_API_URL}?search_query={quote(query)}&start=0&max_results=15"
        response = _request_with_retries(url)
        root = ET.fromstring(response.text)
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            identifier = entry.findtext("{http://www.w3.org/2005/Atom}id", default="")
            if identifier in seen_ids:
                continue
            seen_ids.add(identifier)
            yield entry


def _search_arxiv_metadata(title: str, authors: List[str]) -> Optional[ArxivMetadata]:
    """Search arXiv for a paper that best matches the given title and authors."""

    lead_author = authors[0] if authors else ""
    author_token = lead_author.split()[0] if lead_author else ""

    queries: List[str] = []
    # Try exact phrase matches first, optionally constrained by the full lead author name.
    if title and lead_author:
        queries.append(f'ti:"{title}" AND au:"{lead_author}"')
    if title:
        queries.append(f'ti:"{title}"')

    # Fall back to conjunctions of individual title tokens which allows the API to
    # match when punctuation or accents differ between the sources.
    tokens = _tokenise_title_for_search(title)
    if tokens:
        token_query = " AND ".join(f"ti:{token}" for token in tokens)
        if author_token:
            queries.append(f"{token_query} AND au:{author_token}")
        queries.append(token_query)

    best_entry: Optional[ET.Element] = None
    best_score = 0.0
    for entry in _iter_arxiv_search_entries(queries):
        title_element = entry.find("{http://www.w3.org/2005/Atom}title")
        candidate_title = title_element.text if title_element is not None else ""
        score = _title_similarity(title, candidate_title)
        if score > best_score:
            best_entry = entry
            best_score = score
            if best_score >= 0.92:
                break

    if best_entry is None or best_score < 0.8:
        return None

    return _parse_arxiv_entry(best_entry)


def _resolve_arxiv_metadata(
    paper: Paper, soup: Optional[BeautifulSoup]
) -> Optional[ArxivMetadata]:
    """Resolve arXiv metadata for a paper using page hints and API lookups."""

    identifier: Optional[str] = None
    if soup is not None:
        identifier = _extract_arxiv_identifier(soup)

    if identifier:
        try:
            metadata = _fetch_arxiv_metadata_by_id(identifier)
        except requests.HTTPError:
            metadata = None
        except requests.RequestException as exc:  # pragma: no cover - network issue
            print(f"Failed to retrieve arXiv metadata for {identifier}: {exc}", file=sys.stderr)
            metadata = None
        if metadata:
            return metadata

    try:
        return _search_arxiv_metadata(paper.title, paper.authors)
    except requests.HTTPError:
        return None
    except requests.RequestException as exc:  # pragma: no cover - network issue
        lead_author = paper.authors[0] if paper.authors else "unknown author"
        print(
            f"Failed to search arXiv for '{paper.title}' ({lead_author}): {exc}",
            file=sys.stderr,
        )
        return None


def _populate_additional_details(paper: Paper, fetch_abstract: bool, fetch_arxiv: bool) -> None:
    """Fetch optional abstract text and arXiv metadata for *paper*."""

    needs_abstract = fetch_abstract and not paper.abstract

    if not (needs_abstract or fetch_arxiv):
        return

    soup: Optional[BeautifulSoup] = None
    try:
        soup = _get_soup(paper.paper_url)
    except requests.HTTPError as exc:
        # Leave a note if we expected to capture the abstract but could not fetch the page.
        if needs_abstract:
            print(
                f"Failed to fetch abstract page {paper.paper_url}: {exc}",
                file=sys.stderr,
            )
    except requests.RequestException as exc:  # pragma: no cover - network issue
        print(f"Failed to fetch abstract page {paper.paper_url}: {exc}", file=sys.stderr)

    if needs_abstract and soup is not None:
        paper.abstract = _extract_abstract_from_soup(soup)

    if fetch_arxiv:
        paper.arxiv = _resolve_arxiv_metadata(paper, soup)


def _parse_neurips_listing(year: int) -> List[Paper]:
    """Parse the listing page for *year* and return paper metadata."""

    listing_url = urljoin(NEURIPS_BASE_URL, NEURIPS_LISTING_PATH_TEMPLATE.format(year=year))
    soup = _get_soup(listing_url)

    papers: List[Paper] = []
    for item in soup.select("ul.paper-list li"):
        link = item.find("a", href=True)
        if not link:
            continue

        match = ABSTRACT_PATH_PATTERN.match(link["href"])
        if not match:
            continue

        section = match.group("section")
        paper_hash = match.group("hash")
        abstract_url = urljoin(NEURIPS_BASE_URL, link["href"])
        pdf_url = urljoin(
            NEURIPS_BASE_URL,
            f"/paper_files/paper/{year}/file/{paper_hash}-Paper-{section}.pdf",
        )

        papers.append(
            Paper(
                year=year,
                title=link.get_text(" ", strip=True),
                authors=_extract_authors(item),
                track=_normalise_track(section, item.get("class")),
                paper_url=abstract_url,
                pdf_url=pdf_url,
                abstract=None,
            )
        )

    return papers


def _parse_acl_listing(year: int, event: str) -> List[Paper]:
    """Parse an ACL Anthology event page for *year* and return paper metadata."""

    event_slug = event.strip().lower()
    if event_slug not in ACL_STYLE_CONFERENCES:
        raise ValueError(f"Unsupported ACL Anthology event '{event}'")
    event_path = ACL_EVENT_PATH_TEMPLATE.format(event=event_slug, year=year)
    listing_url = urljoin(ACL_BASE_URL, event_path)
    soup = _get_soup(listing_url)

    papers: List[Paper] = []
    section_selector = f"div[id^='{year}{event_slug}-']"
    for section in soup.select(section_selector):
        header_link = None
        header = section.find("h4")
        if header:
            header_links = header.find_all("a")
            if header_links:
                header_link = header_links[-1]
        track = (
            header_link.get_text(" ", strip=True)
            if header_link
            else section.get("id", event_slug.upper())
        )

        for entry in section.find_all("p", class_="d-sm-flex align-items-stretch"):
            spans = entry.find_all("span")
            if len(spans) < 2:
                continue

            content_span = spans[-1]
            title_container = content_span.find("strong")
            if title_container is None:
                continue
            title_anchor = title_container.find("a", href=True)
            if title_anchor is None:
                continue

            title = title_anchor.get_text(" ", strip=True)
            paper_url = urljoin(ACL_BASE_URL, title_anchor["href"])

            pdf_url: Optional[str] = None
            button_span = entry.find("span", class_="list-button-row")
            if button_span:
                for anchor in button_span.find_all("a", href=True):
                    text = anchor.get_text(" ", strip=True).lower()
                    href = anchor["href"].strip()
                    if "pdf" in text or href.lower().endswith(".pdf"):
                        pdf_url = urljoin(ACL_BASE_URL, href)
                        break
            if not pdf_url:
                pdf_url = f"{paper_url.rstrip('/')}.pdf"

            author_links = [
                anchor
                for anchor in content_span.find_all("a", href=True)
                if anchor is not title_anchor and "/people/" in anchor["href"]
            ]
            authors = [anchor.get_text(" ", strip=True) for anchor in author_links]

            papers.append(
                Paper(
                    year=year,
                    title=title,
                    authors=authors,
                    track=track,
                    paper_url=paper_url,
                    pdf_url=pdf_url,
                    abstract=None,
                )
            )

    return papers


def _iter_openreview_notes(invitation: str) -> Iterable[dict[str, Any]]:
    """Yield OpenReview notes for *invitation* using paginated requests."""

    offset = 0
    limit = 200

    while True:
        params = {
            "invitation": invitation,
            "offset": str(offset),
            "limit": str(limit),
        }
        response = _request_with_retries(OPENREVIEW_API_NOTES_URL, params=params)
        payload = response.json()
        notes = payload.get("notes", [])
        if not notes:
            break
        for note in notes:
            yield note
        offset += len(notes)
        if len(notes) < limit:
            break


def _search_openreview_notes(
    query: str, *, sort: str = "tmdate:asc", page_size: int = 200
) -> Iterable[dict[str, Any]]:
    """Yield OpenReview notes that satisfy a Lucene *query* string."""

    offset = 0
    while True:
        payload = {
            "query": query,
            "limit": page_size,
            "offset": offset,
            "details": "all",
            "sort": sort,
        }
        response = _request_with_retries(
            OPENREVIEW_API_NOTES_SEARCH_URL,
            method="POST",
            json_body=payload,
        )
        data = response.json()
        notes = data.get("notes", [])
        if not notes:
            break
        for note in notes:
            yield note
        offset += len(notes)
        if offset >= data.get("count", 0):
            break


def _normalise_iclr_track(venue: str, year: int) -> str:
    """Extract a human readable track label from an ICLR venue string."""

    if not venue:
        return "Conference"

    prefix = f"ICLR {year}"
    lower_venue = venue.lower()
    if lower_venue.startswith(prefix.lower()):
        suffix = venue[len(prefix) :].strip()
        suffix = suffix.strip("- ")
        if suffix.startswith("(") and suffix.endswith(")"):
            suffix = suffix[1:-1].strip()
        if suffix:
            return suffix.title()
        return "Conference"

    return venue


def _iclr_track_from_decision(decision: str) -> str:
    """Derive a track label from an ICLR decision string."""

    if not decision:
        return "Conference"

    match = re.search(r"accept\s*\(([^)]+)\)", decision, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace("_", " ").strip().title() or "Conference"

    if "accept" in decision.lower():
        return "Conference"

    return decision.strip().title() or "Conference"


def _build_iclr_decision_map(year: int) -> Dict[str, str]:
    """Return a mapping from forum IDs to decision strings for ICLR."""

    invitation = f"ICLR.cc/{year}/Conference/Paper.*"
    decisions: Dict[str, str] = {}
    for note in _iter_openreview_notes(invitation):
        invitation_name = note.get("invitation", "")
        if not invitation_name.endswith("/-/Decision"):
            continue

        forum = note.get("forum")
        if not forum:
            continue

        content = note.get("content", {})
        decision = (content.get("decision") or "").strip()
        if decision:
            decisions[forum] = decision

    return decisions


def _iclr_note_to_paper(
    year: int, note: dict[str, Any], *, decision_map: Dict[str, str]
) -> Optional[Paper]:
    """Convert an OpenReview note into a :class:`Paper` if it is accepted."""

    content = note.get("content", {})
    venue = (content.get("venue") or "").strip()
    decision_text = (content.get("decision") or "").strip()

    title = content.get("title", "").strip()
    authors = [author.strip() for author in content.get("authors", []) if author.strip()]
    forum = note.get("forum") or note.get("id")
    if not forum or not title:
        return None

    pdf_path = content.get("pdf") or ""
    pdf_url = urljoin(OPENREVIEW_BASE_URL, pdf_path)
    paper_url = urljoin(OPENREVIEW_BASE_URL, f"/forum?id={forum}")

    decision = decision_text
    if not venue:
        if not decision:
            decision = decision_map.get(forum)

        status = decision or ""
    else:
        status = venue

    normalised = status.lower()
    if "submitted" in normalised or "reject" in normalised or "withdraw" in normalised:
        return None

    if venue:
        track = _normalise_iclr_track(venue, year)
    else:
        track = _iclr_track_from_decision(decision or "")
    abstract = (content.get("abstract") or "").strip() or None

    return Paper(
        year=year,
        title=title,
        authors=authors,
        track=track,
        paper_url=paper_url,
        pdf_url=pdf_url,
        abstract=abstract,
    )


def _parse_iclr_listing(year: int) -> List[Paper]:
    """Fetch accepted ICLR papers for *year* from OpenReview."""

    invitation = f"ICLR.cc/{year}/Conference/-/Blind_Submission"
    notes = list(_iter_openreview_notes(invitation))
    notes.sort(key=lambda note: note.get("number", 0))

    papers: List[Paper] = []
    decision_map: Optional[Dict[str, str]] = None
    for note in notes:
        if decision_map is None:
            content = note.get("content", {})
            venue_present = (content.get("venue") or "").strip()
            decision_inline = (content.get("decision") or "").strip()
            if not venue_present and not decision_inline:
                decision_map = _build_iclr_decision_map(year)

        paper = _iclr_note_to_paper(year, note, decision_map=decision_map or {})
        if paper is not None:
            papers.append(paper)

    if not papers:
        search_queries = [f'content.venueid:"ICLR.cc/{year}/Conference"']
        for track in ("Poster", "Spotlight", "Oral"):
            search_queries.append(
                f'content.venueid:"ICLR.cc/{year}/Conference/{track}"'
            )
            search_queries.append(f'content.venue:"ICLR {year} {track}"')
        search_queries.append(f'content.venue:"ICLR {year}"')

        seen_forums: Dict[str, Paper] = {}
        for query in search_queries:
            for note in _search_openreview_notes(query):
                forum = note.get("forum") or note.get("id")
                if not forum or forum in seen_forums:
                    continue
                paper = _iclr_note_to_paper(year, note, decision_map={})
                if paper is not None:
                    seen_forums[forum] = paper

        papers = list(seen_forums.values())

    if not papers:
        raise ValueError(
            "No accepted ICLR submissions were found via the published OpenReview feeds. "
            "Check the year or try again later."
        )

    return papers


def _render_progress(
    completed: int,
    total: int,
    *,
    prefix: str = "",
    end: str = "\n",
) -> None:
    """Render a simple textual progress indicator to ``stderr``."""

    if total <= 0:
        return

    percent = int((completed / total) * 100)
    bar_length = 20
    filled = int(bar_length * completed / total)
    bar = "#" * filled + "-" * (bar_length - filled)

    sys.stderr.write(
        f"\r{prefix}: [{bar}] {completed}/{total} ({percent:3d}%)"
    )
    if completed >= total:
        sys.stderr.write(end)
    sys.stderr.flush()


def scrape(
    year: int,
    conference: str = "neurips",
    limit: Optional[int] = None,
    fetch_abstracts: bool = True,
    fetch_arxiv: bool = True,
    show_progress: bool = False,
) -> List[Paper]:
    """Scrape conference papers for *year*.

    Parameters
    ----------
    year:
        The conference year to fetch.
    conference:
        Which conference to scrape. Supported values are ``"neurips"``,
        ``"iclr"``, ``"acl"``, ``"naacl"``, and ``"emnlp"``.
    limit:
        Optional maximum number of papers to return. The papers appear in the
        same order as listed on the website.
    fetch_abstracts:
        Whether to fetch abstract text for each paper.
    fetch_arxiv:
        Whether to attempt to enrich papers with arXiv metadata.
    show_progress:
        Whether to display a progress indicator while fetching optional metadata.
    """

    normalised_conference = conference.strip().lower()

    if normalised_conference == "neurips":
        papers = _parse_neurips_listing(year)
    elif normalised_conference == "iclr":
        papers = _parse_iclr_listing(year)
    elif normalised_conference in ACL_STYLE_CONFERENCES:
        papers = _parse_acl_listing(year, normalised_conference)
    else:
        raise ValueError(f"Unsupported conference '{conference}'")
    if limit is not None:
        papers = papers[:limit]

    if fetch_abstracts or fetch_arxiv:
        total = len(papers)
        use_progress = show_progress and sys.stderr.isatty()
        for index, paper in enumerate(papers, start=1):
            _populate_additional_details(paper, fetch_abstracts, fetch_arxiv)
            if use_progress:
                _render_progress(index, total, prefix="Fetching details", end="\n")
        if use_progress:
            sys.stderr.flush()

    return papers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape accepted conference papers")
    parser.add_argument("--year", type=int, required=True, help="Conference year to scrape")
    parser.add_argument(
        "--conference",
        type=str,
        default="neurips",
        choices=["neurips", "iclr", "acl", "naacl", "emnlp"],
        help="Which conference to scrape (default: neurips)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of papers to fetch (useful for testing)",
    )
    parser.add_argument(
        "--skip-abstracts",
        action="store_true",
        help="Skip fetching abstract text to reduce the number of requests",
    )
    parser.add_argument(
        "--skip-arxiv",
        action="store_true",
        help="Skip attempting to link arXiv entries for each paper",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress indicator while fetching paper details",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="Where to write the JSON output (default: stdout)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        papers = scrape(
            args.year,
            conference=args.conference,
            limit=args.limit,
            fetch_abstracts=not args.skip_abstracts,
            fetch_arxiv=not args.skip_arxiv,
            show_progress=not args.no_progress,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except requests.HTTPError as exc:
        print(
            f"Failed to fetch {args.conference.upper()} {args.year} listing: {exc}",
            file=sys.stderr,
        )
        return 1
    except requests.RequestException as exc:  # pragma: no cover - network issue
        print(
            f"Network error while fetching {args.conference.upper()} data: {exc}",
            file=sys.stderr,
        )
        return 1

    data = [asdict(paper) for paper in papers]
    output = json.dumps(data, indent=2, ensure_ascii=False)

    if args.output == "-":
        print(output)
    else:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
