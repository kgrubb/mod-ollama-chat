CREATE TABLE IF NOT EXISTS mod_ollama_chat_history (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    bot_guid BIGINT UNSIGNED NOT NULL,
    player_guid BIGINT UNSIGNED NOT NULL,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    player_message TEXT NOT NULL,
    bot_reply TEXT NOT NULL,
    UNIQUE KEY unique_history (bot_guid, player_guid, player_message(255), bot_reply(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS mod_ollama_chat_memory_semantic (
    bot_guid BIGINT UNSIGNED NOT NULL,
    player_guid BIGINT UNSIGNED NOT NULL,
    player_name VARCHAR(12) NOT NULL DEFAULT '',
    memory_text TEXT NOT NULL DEFAULT '',
    facts_text TEXT NOT NULL DEFAULT '',
    notes_text TEXT NOT NULL DEFAULT '',
    last_player_at DATETIME NULL,
    compacted_turn_count INT UNSIGNED NOT NULL DEFAULT 0,
    last_compacted_at DATETIME NULL,
    PRIMARY KEY (bot_guid, player_guid)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS mod_ollama_chat_memory_episodic (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    bot_guid BIGINT UNSIGNED NOT NULL,
    player_guid BIGINT UNSIGNED NOT NULL,
    player_name VARCHAR(12) NOT NULL DEFAULT '',
    context TEXT NOT NULL,
    bot_reply TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_episodic_pair (bot_guid, player_guid, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS mod_ollama_chat_memory_archive (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    bot_guid BIGINT UNSIGNED NOT NULL,
    player_guid BIGINT UNSIGNED NOT NULL,
    context TEXT NOT NULL,
    bot_reply TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_archive_pair_time (bot_guid, player_guid, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
