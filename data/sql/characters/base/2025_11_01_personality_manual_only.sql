-- Add manual_only column to personality templates table (idempotent for re-applied base SQL)
SET @ollama_manual_only_exists = (
    SELECT COUNT(*)
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'mod_ollama_chat_personality_templates'
      AND COLUMN_NAME = 'manual_only'
);
SET @ollama_manual_only_sql = IF(
    @ollama_manual_only_exists = 0,
    'ALTER TABLE `mod_ollama_chat_personality_templates` ADD COLUMN `manual_only` TINYINT(1) NOT NULL DEFAULT 0 AFTER `prompt`',
    'SELECT 1'
);
PREPARE ollama_manual_only_stmt FROM @ollama_manual_only_sql;
EXECUTE ollama_manual_only_stmt;
DEALLOCATE PREPARE ollama_manual_only_stmt;
