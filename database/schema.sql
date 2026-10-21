-- CircuitVision SQLite reference catalog. Seeded from the existing team-written
-- component guide; object-detection predictions do not prove specifications.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS component_categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    purpose TEXT NOT NULL,
    learning_notes TEXT NOT NULL,
    value_guidance TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS components (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL COLLATE NOCASE UNIQUE,
    category_id INTEGER NOT NULL REFERENCES component_categories(id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    information_level TEXT NOT NULL DEFAULT 'category'
        CHECK(information_level IN ('category', 'general')),
    last_updated TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_registry (
    code TEXT PRIMARY KEY,
    display_name TEXT NOT NULL UNIQUE,
    intended_class_count INTEGER NOT NULL CHECK(intended_class_count > 0)
);

CREATE TABLE IF NOT EXISTS model_classes (
    model_code TEXT NOT NULL REFERENCES model_registry(code)
        ON DELETE CASCADE,
    class_id INTEGER NOT NULL CHECK(class_id >= 0),
    component_id INTEGER NOT NULL REFERENCES components(id)
        ON DELETE RESTRICT,
    PRIMARY KEY(model_code, class_id)
);

CREATE INDEX IF NOT EXISTS idx_components_category ON components(category_id);
CREATE INDEX IF NOT EXISTS idx_model_classes_component ON model_classes(component_id);
