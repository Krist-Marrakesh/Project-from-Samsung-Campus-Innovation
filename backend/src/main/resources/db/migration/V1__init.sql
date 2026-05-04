-- fractalov backend Stage 3 baseline schema.
-- Three aggregates: project (top-level container) → recipe (saved fractal config)
-- → render (materialised image + metadata + filesystem path).

CREATE TABLE IF NOT EXISTS projects (
    id          UUID PRIMARY KEY,
    name        VARCHAR(200)               NOT NULL,
    description TEXT,
    owner_id    VARCHAR(100)               NOT NULL DEFAULT 'anonymous',
    created_at  TIMESTAMP WITH TIME ZONE   NOT NULL DEFAULT now(),
    updated_at  TIMESTAMP WITH TIME ZONE   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects (owner_id, created_at DESC);

CREATE TABLE IF NOT EXISTS recipes (
    id           UUID PRIMARY KEY,
    project_id   UUID                       NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    name         VARCHAR(200),
    fractal_type VARCHAR(40)                NOT NULL,
    recipe_json  JSONB                      NOT NULL,
    version      INT                        NOT NULL DEFAULT 1,
    created_at   TIMESTAMP WITH TIME ZONE   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_recipes_project ON recipes (project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recipes_fractal_type ON recipes (fractal_type);

CREATE TABLE IF NOT EXISTS renders (
    id            UUID PRIMARY KEY,
    recipe_id     UUID                       NOT NULL REFERENCES recipes (id) ON DELETE CASCADE,
    image_path    VARCHAR(500)               NOT NULL,
    width_px      INT                        NOT NULL,
    height_px     INT                        NOT NULL,
    palette_name  VARCHAR(80)                NOT NULL,
    color_mode    VARCHAR(20)                NOT NULL,
    samples_per_axis SMALLINT                NOT NULL DEFAULT 1,
    render_ms     BIGINT                     NOT NULL,
    colorize_ms   BIGINT                     NOT NULL,
    encode_ms     BIGINT                     NOT NULL,
    total_ms      BIGINT                     NOT NULL,
    file_size_bytes BIGINT                   NOT NULL,
    created_at    TIMESTAMP WITH TIME ZONE   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_renders_recipe ON renders (recipe_id, created_at DESC);
