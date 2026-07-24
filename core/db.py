from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from core.config import get_settings


class Base(DeclarativeBase):
    pass


def get_engine():
    return create_engine(get_settings().database_url)


def init_db(engine=None) -> None:
    """Legt die pgvector-Extension und alle Tabellen an (idempotent).

    Base.metadata.create_all() legt nur komplett fehlende Tabellen/Typen an --
    bei bereits existierenden Tabellen/Enums (wie hier, aus Phase 1) werden
    neue Spalten oder Enum-Werte NICHT automatisch nachgezogen. Die
    ALTER-Statements unten holen das idempotent nach (IF NOT EXISTS).
    """
    import core.models  # noqa: F401  registriert alle Modelle auf Base

    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)

    # Phase 2: EntityType um chemical/cell_line/species erweitert, Entity um
    # ensembl_id/uniprot_id -- beides additiv auf bereits existierenden
    # Tabellen/Typen aus Phase 1, entity/mention waren zu dem Zeitpunkt leer.
    for value in ("chemical", "cell_line", "species"):
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TYPE entitytype ADD VALUE IF NOT EXISTS '{value}'"))
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE entity ADD COLUMN IF NOT EXISTS ensembl_id VARCHAR"))
        conn.execute(text("ALTER TABLE entity ADD COLUMN IF NOT EXISTS uniprot_id VARCHAR"))


def get_sessionmaker(engine=None):
    return sessionmaker(bind=engine or get_engine())
