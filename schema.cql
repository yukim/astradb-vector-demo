CREATE TABLE IF NOT EXISTS query_results (
    query text,
    pattern text,
    rank int,
    id text,
    relevant boolean,
    PRIMARY KEY (query, pattern, rank, id)
);

BEGIN BATCH
    INSERT INTO query_results (query, pattern, rank, id, relevant) VALUES (?, ?, ?, ?, ?);
APPLY BATCH;