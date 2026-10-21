"""CircuitVision's relational SQLite catalog for YOLO component reference data.

Seed originates in our existing component_details.mjs guide and trained
87-class map. No inference prediction is an electrical measurement.

Set CV_DATABASE_PATH to an absolute, writable .sqlite3 path if needed.
The default /tmp database on Render's ephemeral filesystem is rebuilt
from version-controlled schema + seed at service startup.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import threading

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "database" / "schema.sql"
SEED = ROOT / "database" / "component_seed.json"
_INIT_LOCK = threading.RLock()
_initialized_at: str | None = None


def database_path() -> Path:
    return Path(os.getenv("CV_DATABASE_PATH", "/tmp/circuitvision_components.sqlite3")).expanduser()


def connection(db_file: Path | None = None) -> sqlite3.Connection:
    path = db_file or database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), timeout=8)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=8000")
    return db


def initialize_database(db_file: Path | None = None) -> Path:
    """Create relational tables, refresh seed metadata, preserve existing IDs.

    The seed's source of truth is checked into git. This is idempotent and safe
    to rerun when Render restarts or when the database is initialized locally.
    """
    global _initialized_at
    path = (db_file or database_path()).resolve()
    with _INIT_LOCK:
        if _initialized_at == str(path) and path.is_file():
            return path
        seed = json.loads(SEED.read_text(encoding="utf-8"))
        with closing(connection(path)) as db:
            db.executescript(SCHEMA.read_text(encoding="utf-8"))
            with db:
                for category in seed["categories"]:
                    db.execute("""
                        INSERT INTO component_categories
                            (slug,title,purpose,learning_notes,value_guidance)
                        VALUES (?,?,?,?,?)
                        ON CONFLICT(slug) DO UPDATE SET
                            title=excluded.title, purpose=excluded.purpose,
                            learning_notes=excluded.learning_notes,
                            value_guidance=excluded.value_guidance
                    """, (category["slug"],category["title"],category["role"],
                          category["learn"],category["value"]))
                for comp in seed["components"]:
                    level = ("general" if comp["category"]=="general" else "category")
                    db.execute("""
                        INSERT INTO components (label,category_id,information_level)
                        SELECT ?,id,? FROM component_categories WHERE slug=?
                        ON CONFLICT(label) DO UPDATE SET
                            category_id=excluded.category_id,
                            information_level=excluded.information_level,
                            last_updated=CURRENT_TIMESTAMP
                    """, (comp["label"],level,comp["category"]))
                for code, display, size in [
                    ("yolov11_1","YOLOv11 (1)",57),
                    ("yolov11_2","YOLOv11 (2)",87),
                ]:
                    db.execute("""
                        INSERT INTO model_registry (code,display_name,intended_class_count)
                        VALUES (?,?,?)
                        ON CONFLICT(code) DO UPDATE SET
                          display_name=excluded.display_name,
                          intended_class_count=excluded.intended_class_count
                    """, (code,display,size))
                for entry in seed["model_classes"]:
                    row=db.execute("SELECT id FROM components WHERE label=? COLLATE NOCASE",
                                   (entry["label"],)).fetchone()
                    if row is None:
                        raise ValueError("Missing seed reference for model class "+entry["label"])
                    db.execute("""
                        INSERT INTO model_classes(model_code,class_id,component_id)
                        VALUES (?,?,?)
                        ON CONFLICT(model_code,class_id)
                        DO UPDATE SET component_id=excluded.component_id
                    """, (entry["model"],entry["class_id"],row["id"]))
                valid=db.execute("PRAGMA foreign_key_check").fetchall()
                if valid:
                    raise ValueError("Foreign-key violation in the component database")
                count=db.execute("SELECT COUNT(*) FROM model_classes WHERE model_code='yolov11_2'").fetchone()[0]
                if count != 87:
                    raise ValueError(f"Expected 87 mapped YOLOv11 (2) classes; found {count}")
        _initialized_at = str(path)
        return path


def _reference_from_row(row: sqlite3.Row) -> dict:
    return {
        "label": row["label"],
        "name": row["category_title"],
        "role": row["purpose"],
        "learn": row["learning_notes"],
        "value": row["value_guidance"],
        "information_level": row["information_level"],
        "category": row["category_slug"],
        "reference_source": "sqlite",
        "note": "Educational reference only; confirm specifications from markings or a datasheet.",
    }


_JOIN = """SELECT c.label,c.information_level,
                  g.slug AS category_slug,g.title AS category_title,
                  g.purpose,g.learning_notes,g.value_guidance
             FROM components c
             JOIN component_categories g ON c.category_id=g.id"""


def get_component(label: str, db_file: Path | None = None) -> dict | None:
    initialize_database(db_file)
    with closing(connection(db_file)) as db:
        row=db.execute(_JOIN+" WHERE c.label=? COLLATE NOCASE",(label,)).fetchone()
        return _reference_from_row(row) if row is not None else None


def lookup_components(labels: list[str], db_file: Path | None = None) -> dict[str, dict]:
    """Batch-fetch all detected class labels in one query per inference request."""
    initialize_database(db_file)
    names=list(dict.fromkeys(str(s).strip() for s in labels if str(s).strip()))
    if not names:
        return {}
    if len(names)>400:
        names=names[:400]
    markers=",".join("?" for _ in names)
    with closing(connection(db_file)) as db:
        rows=db.execute(_JOIN+f" WHERE c.label IN ({markers})",names).fetchall()
    return {row["label"].casefold(): _reference_from_row(row) for row in rows}


def list_components(query: str = "", limit: int = 100, db_file: Path | None = None) -> list[dict]:
    initialize_database(db_file)
    limit=max(1,min(150,int(limit)))
    pattern="%"+query.strip().replace("%",r"\%").replace("_",r"\_")+"%"
    with closing(connection(db_file)) as db:
        rows=db.execute(_JOIN+
            " WHERE c.label LIKE ? ESCAPE '\\' ORDER BY c.label LIMIT ?",
            (pattern,limit)).fetchall()
    return [_reference_from_row(row) for row in rows]


def database_status(db_file: Path | None = None) -> dict:
    initialize_database(db_file)
    with closing(connection(db_file)) as db:
        categories=db.execute("SELECT COUNT(*) FROM component_categories").fetchone()[0]
        components=db.execute("SELECT COUNT(*) FROM components").fetchone()[0]
        mapped=db.execute("SELECT COUNT(*) FROM model_classes WHERE model_code='yolov11_2'").fetchone()[0]
        foreign_keys_ok=not bool(db.execute("PRAGMA foreign_key_check").fetchall())
    return {"ready":True, "categories":categories,"components":components,
            "mapped_model_2_classes":mapped,"foreign_keys_valid":foreign_keys_ok}


def upsert_component(label: str, category: str, db_file: Path | None = None) -> dict:
    """Maintain a catalog entry via trusted CLI, never through a public write API."""
    label=label.strip()
    if not label or len(label)>120:
        raise ValueError("Component label must contain 1–120 characters")
    initialize_database(db_file)
    with closing(connection(db_file)) as db:
        with db:
            row=db.execute("SELECT id FROM component_categories WHERE slug=?",(category,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown category: {category}")
            db.execute("""
                INSERT INTO components(label,category_id,information_level)
                VALUES (?,?,?)
                ON CONFLICT(label) DO UPDATE SET
                  category_id=excluded.category_id,
                  information_level=excluded.information_level,
                  last_updated=CURRENT_TIMESTAMP
            """,(label,row["id"],"general" if category=="general" else "category"))
    return get_component(label,db_file)


def main() -> None:
    parser=argparse.ArgumentParser(description="CircuitVision SQLite reference catalog")
    parser.add_argument("command",choices=["init","stats","list","upsert"])
    parser.add_argument("--label",default="")
    parser.add_argument("--category",default="")
    parser.add_argument("--query",default="")
    args=parser.parse_args()
    if args.command=="init":
        print("Initialized",initialize_database())
    elif args.command=="stats":
        print(json.dumps(database_status(),indent=2))
    elif args.command=="list":
        print(json.dumps(list_components(args.query),indent=2))
    else:
        if not args.label or not args.category:
            parser.error("upsert requires --label and --category")
        print(json.dumps(upsert_component(args.label,args.category),indent=2))


if __name__=="__main__":
    main()
