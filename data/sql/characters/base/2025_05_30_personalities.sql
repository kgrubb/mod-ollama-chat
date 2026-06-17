CREATE TABLE IF NOT EXISTS `mod_ollama_chat_personality` (
  `guid` BIGINT NOT NULL,
  `personality` VARCHAR(64) NOT NULL,
  PRIMARY KEY (`guid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Upgrade legacy int schema without dropping assignments (no-op when already migrated)
SET @ollama_personality_guid_type = (
    SELECT DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'mod_ollama_chat_personality'
      AND COLUMN_NAME = 'guid'
    LIMIT 1
);
SET @ollama_personality_migrate_sql = IF(
    @ollama_personality_guid_type = 'int',
    'ALTER TABLE `mod_ollama_chat_personality`
        MODIFY COLUMN `guid` BIGINT NOT NULL,
        MODIFY COLUMN `personality` VARCHAR(64) NOT NULL',
    'SELECT 1'
);
PREPARE ollama_personality_migrate_stmt FROM @ollama_personality_migrate_sql;
EXECUTE ollama_personality_migrate_stmt;
DEALLOCATE PREPARE ollama_personality_migrate_stmt;
