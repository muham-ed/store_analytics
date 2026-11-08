-- PostgreSQL Schema for Store Night Camera Analytics Platform

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Cameras table storing region configurations
CREATE TABLE IF NOT EXISTS cameras (
    camera_id VARCHAR(50) PRIMARY KEY,
    store_id VARCHAR(50) NOT NULL,
    camera_name VARCHAR(100) NOT NULL,
    config_json JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Video Tasks table for Distributed Worker Execution with Resilience
CREATE TABLE IF NOT EXISTS video_tasks (
    task_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id VARCHAR(50) REFERENCES cameras(camera_id) ON DELETE CASCADE,
    video_path TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING', -- PENDING, PROCESSING, COMPLETED, FAILED
    locked_by VARCHAR(100) DEFAULT NULL,
    locked_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    heartbeat_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    attempts INT DEFAULT 0,
    max_attempts INT DEFAULT 3,
    error_message TEXT DEFAULT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    completed_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_video_tasks_status ON video_tasks(status);

-- Scenario Analytics Output Results Table with UPSERT constraint
CREATE TABLE IF NOT EXISTS scenario_results (
    result_id BIGSERIAL PRIMARY KEY,
    task_id UUID REFERENCES video_tasks(task_id) ON DELETE CASCADE,
    camera_id VARCHAR(50) NOT NULL,
    scenario_name VARCHAR(50) NOT NULL,
    metrics_json JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_task_scenario UNIQUE (task_id, scenario_name)
);

CREATE INDEX IF NOT EXISTS idx_scenario_results_camera_scenario ON scenario_results(camera_id, scenario_name);
CREATE INDEX IF NOT EXISTS idx_scenario_results_task ON scenario_results(task_id);
