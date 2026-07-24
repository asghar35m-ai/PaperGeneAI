from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.db import get_sessionmaker
from core.models import Document, JobRun, JobStage, JobStatus
from ingest.europepmc import EuropePmcArticle, EuropePmcClient
from ingest.query_strategy import DEFAULT_QUERIES_PATH, load_queries


def _dedup_key(article: EuropePmcArticle) -> tuple[str, str]:
    if article.pmid:
        return ("pmid", article.pmid)
    return ("doi", article.doi.lower())  # article ohne PMID *und* DOI wird vorher aussortiert


def insert_if_new(session: Session, article: EuropePmcArticle) -> bool:
    """Schreibt einen Artikel als documents-Zeile plus job_runs-Eintrag,
    ausser er existiert schon (per PMID, sonst DOI). Gibt zurueck, ob neu
    eingefuegt wurde. Von ingest/run.py und ingest/seed.py gemeinsam genutzt."""
    if article.pmid:
        existing = session.scalar(select(Document).where(Document.pmid == article.pmid))
    else:
        existing = session.scalar(select(Document).where(Document.doi == article.doi))

    if existing:
        return False

    document = Document(
        pmid=article.pmid,
        doi=article.doi,
        title=article.title,
        abstract=article.abstract,
        source=article.source,
        pub_year=article.pub_year,
        is_open_access=article.is_open_access,
        full_text_url=article.full_text_url,
    )
    session.add(document)
    session.flush()  # document.id fuer den JobRun

    now = datetime.now(timezone.utc)
    session.add(
        JobRun(
            document_id=document.id,
            stage=JobStage.INGEST,
            status=JobStatus.DONE,
            started_at=now,
            finished_at=now,
        )
    )
    session.commit()
    return True


@dataclass
class FetchResult:
    articles: dict[tuple[str, str], EuropePmcArticle]
    skipped_no_identifier: int


def fetch_unique_articles(queries_path: Path = DEFAULT_QUERIES_PATH) -> FetchResult:
    """Fuehrt alle konfigurierten Anfragen aus und dedupliziert per PMID/DOI.

    Artikel ganz ohne PMID und DOI (z.B. Konferenz-Abstract-Sammelbaende wie
    "Posters" oder "UEG Week 2023 Moderated Posters") werden aussortiert --
    CLAUDE.md verlangt fuer jede Aussage eine PMID-Rueckfuehrbarkeit, und ohne
    jeden stabilen Identifier waere das nie moeglich. DOI-only bleibt erlaubt
    (z.B. spaeter bioRxiv-Preprints ohne PMID).
    """
    client = EuropePmcClient()
    queries = load_queries(queries_path)

    unique: dict[tuple[str, str], EuropePmcArticle] = {}
    skipped_no_identifier = 0
    for q in queries:
        hit_count = client.hit_count(q.query)
        for article in client.search(q.query, page_size=100, max_results=hit_count):
            if not article.pmid and not article.doi:
                skipped_no_identifier += 1
                continue
            unique.setdefault(_dedup_key(article), article)

    return FetchResult(articles=unique, skipped_no_identifier=skipped_no_identifier)


@dataclass
class IngestionSummary:
    fetched: int
    inserted: int
    skipped_existing: int
    skipped_no_identifier: int


def run_ingestion(queries_path: Path = DEFAULT_QUERIES_PATH) -> IngestionSummary:
    """Schreibt alle eindeutigen Artikel als documents-Zeilen, idempotent:
    bereits vorhandene (per PMID/DOI) werden uebersprungen statt dupliziert."""
    fetch_result = fetch_unique_articles(queries_path)

    session_factory = get_sessionmaker()
    inserted = 0
    skipped = 0

    with session_factory() as session:
        for article in fetch_result.articles.values():
            if insert_if_new(session, article):
                inserted += 1
            else:
                skipped += 1

    return IngestionSummary(
        fetched=len(fetch_result.articles),
        inserted=inserted,
        skipped_existing=skipped,
        skipped_no_identifier=fetch_result.skipped_no_identifier,
    )


if __name__ == "__main__":
    summary = run_ingestion()
    print(f"Eindeutige Artikel gesamt:        {summary.fetched}")
    print(f"Neu eingefuegt:                   {summary.inserted}")
    print(f"Bereits vorhanden:                {summary.skipped_existing}")
    print(f"Uebersprungen (weder PMID/DOI):   {summary.skipped_no_identifier}")
