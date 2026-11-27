from collections.abc import Iterator

from sqlmodel import Session, SQLModel, create_engine

from . import config

engine = create_engine(f"sqlite:///{config.DB_PATH}", connect_args={"check_same_thread": False})


def init_db() -> None:
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    from . import tables  # noqa: F401  (registers tables on the metadata)

    SQLModel.metadata.create_all(engine)
    _migrate()


def _migrate() -> None:
    """Add columns introduced after a database was created (create_all does not alter tables)."""
    with engine.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(annotation)")}
        if "attributes" not in cols:
            conn.exec_driver_sql("ALTER TABLE annotation ADD COLUMN attributes JSON NOT NULL DEFAULT '{}'")


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
