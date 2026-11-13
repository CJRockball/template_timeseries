-- extract_wafers.sql
-- Full wide frame: sensor readings joined to labels, chronological.
-- Params: start, end (datetime, inclusive); cutoff, exclude_after
--         (datetime or NULL; NULL cutoff -> split = 'unassigned',
--          NULL exclude_after -> no 'excluded' rows)
-- Returns: wafer_id, s001..s590, target, timestamp, split
-- Convention: cv is ts < cutoff; holdout is cutoff <= ts < exclude_after.
-- Callers must assert no timestamp equals cutoff exactly.
SELECT r.*, l.target, l.timestamp,
    CASE
        WHEN :exclude_after IS NOT NULL AND l.timestamp >= :exclude_after THEN 'excluded'
        WHEN :cutoff IS NULL THEN 'unassigned'
        WHEN l.timestamp < :cutoff THEN 'cv'
        ELSE 'holdout'
    END AS split
FROM sensor_readings r
JOIN wafer_labels l USING (wafer_id)
WHERE l.timestamp BETWEEN :start AND :end
ORDER BY l.timestamp, r.wafer_id;