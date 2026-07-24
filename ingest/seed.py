from dataclasses import dataclass
from pathlib import Path

import yaml

from core.db import get_sessionmaker
from ingest.europepmc import EuropePmcArticle, EuropePmcClient
from ingest.run import insert_if_new

DEFAULT_SEED_PATH = Path("config/seed_pmids.yaml")


@dataclass(frozen=True)
class SeedPmid:
    pmid: str
    label: str


def load_seed_pmids(path: Path = DEFAULT_SEED_PATH) -> list[SeedPmid]:
    data = yaml.safe_load(path.read_text())
    return [SeedPmid(**entry) for entry in data["pmids"]]


def fetch_seed_articles(path: Path = DEFAULT_SEED_PATH) -> dict[str, EuropePmcArticle]:
    """Holt jede Seed-PMID direkt per PMID (nicht ueber eine Themen-Anfrage)."""
    client = EuropePmcClient()
    seeds = load_seed_pmids(path)

    articles: dict[str, EuropePmcArticle] = {}
    for seed in seeds:
        results = list(
            client.search(f"EXT_ID:{seed.pmid} AND SRC:MED", page_size=1, max_results=1)
        )
        if not results:
            print(f"WARNUNG: PMID {seed.pmid} ({seed.label}) bei Europe PMC nicht gefunden")
            continue
        articles[seed.pmid] = results[0]

    return articles


@dataclass
class SeedIngestionSummary:
    fetched: int
    inserted: int
    skipped_existing: int


def run_seed_ingestion(path: Path = DEFAULT_SEED_PATH) -> SeedIngestionSummary:
    articles = fetch_seed_articles(path)

    session_factory = get_sessionmaker()
    inserted = 0
    skipped = 0

    with session_factory() as session:
        for article in articles.values():
            if insert_if_new(session, article):
                inserted += 1
            else:
                skipped += 1

    return SeedIngestionSummary(
        fetched=len(articles), inserted=inserted, skipped_existing=skipped
    )


if __name__ == "__main__":
    summary = run_seed_ingestion()
    print(f"Seed-PMIDs gefunden: {summary.fetched}")
    print(f"Neu eingefuegt:      {summary.inserted}")
    print(f"Bereits vorhanden:   {summary.skipped_existing}")
