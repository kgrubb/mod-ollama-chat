# WoW RAG Sample Data

This directory contains sample JSON data files for the Ollama Chat RAG (Retrieval-Augmented Generation) system. These files provide comprehensive World of Warcraft knowledge that the bot can use to enhance its responses with relevant game information.

## File Structure

Each JSON file contains an array of knowledge entries with the following structure:
```json
{
  "id": "unique_identifier",
  "title": "Entry Title",
  "content": "Detailed description and information",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "tags": ["category1", "category2", "category3"]
}
```

## Available Data Files

### wow_professions.json
Contains information about all crafting and gathering professions in World of Warcraft:
- Blacksmithing, Leatherworking, Tailoring, Engineering, Alchemy
- Enchanting, Jewelcrafting, Inscription, Herbalism, Mining
- Skinning, Fishing, Cooking
- Profession specializations and leveling strategies

### wow_classes_factions.json
Covers character classes and faction information:
- All 9 classes (Warrior, Paladin, Hunter, Rogue, Priest, Shaman, Mage, Warlock, Druid)
- Class specializations and playstyles
- Alliance and Horde faction details
- Racial abilities and faction-specific content

### wow_zones.json
Starting zones, classic and Northrend leveling zones, and major cities.

### wow_outland.json
Outland zones (TBC, level 58-70).

### wow_dungeons_raids.json
Classic, TBC, and Wrath dungeons and raids.

### wow_quest_pois.json
Quest locations, zone/subzone landmarks, and dungeon entrances for location-aware RAG ("where is…?" queries).

### wow_quest_npcs.json
Quest-giver NPCs by zone (from world DB export). Tune or omit this file separately if NPC hits crowd out location results.

### wow_mechanics.json
Core game systems; includes Wrath additions (dual spec, LFG, emblems, cold weather flying).

### wow_items_equipment.json
Equipment and item information:
- Tier sets and raid gear progression
- Legendary weapons and their acquisition
- Profession tools and enhancements
- Mount and pet collections
- Enchantments, gems, and socketed items
- Consumables, trade goods, keys
- Cosmetic items and relics

### wow_npcs_creatures.json
NPC and creature information:
- World bosses and rare elites
- Quest givers and class trainers
- Vendors, flight masters, innkeepers
- Bankers, auctioneers, stable masters
- Important lore figures and faction leaders
- Guards, transportation NPCs

### wow_pvp.json
Player vs Player content:
- Battlegrounds (Warsong Gulch, Arathi Basin, Alterac Valley)
- Arena combat (2v2, 3v3, 5v5)
- World PvP and ganking
- PvP ranks, titles, and gear progression
- Honor system and rewards
- Strategies, team composition, psychology

### wow_quests_storylines.json
Quest and narrative content:
- Class-specific storylines
- Zone story arcs and progression
- Reputation grind quests
- Dungeon and raid quest chains
- World event quests
- Epic questlines and profession quests
- Faction conflicts and lore quests
- Various quest types (escort, collection, timed, group)

### wow_general_tips.json
General gameplay knowledge and tips:
- Leveling strategies and efficiency
- Gold making methods and economy
- UI customization and addons
- Server types and character creation
- Time management and communication
- Gear optimization and playstyle adaptation
- Burnout prevention and achievement hunting
- Alt management and endgame content

## Usage

1. **Loading Data**: The RAG system automatically loads all JSON files from this directory on startup
2. **Query Processing**: When a player message is analyzed, the system searches for relevant entries based on keywords and content similarity
3. **Response Enhancement**: Relevant information is added to the bot's prompt to provide contextually appropriate WoW knowledge
4. **Expansion**: Add new JSON files following the same structure to extend the knowledge base

## Customization

### Adding New Knowledge
Create new JSON files with the same structure to add more WoW information. Each entry should have:
- Unique `id` for identification
- Descriptive `title` for the topic
- Comprehensive `content` with detailed information
- Relevant `keywords` for search matching
- Appropriate `tags` for categorization

### Modifying Existing Data
Edit existing JSON files to update information, add new entries, or correct inaccuracies. The system will reload data on the next server restart.

### Best Practices
- Keep content accurate and up-to-date with Wrath of the Lich King
- Use descriptive keywords that players might use in chat
- Include level ranges, locations, and requirements where relevant
- Focus on Wrath of the Lich King content specifically
- Maintain consistent formatting and structure

## Regenerating the corpus

Edit JSON files directly or rebuild `wow_quest_pois.json` and `wow_quest_npcs.json` from world DB locally. Commit JSON only, not builder tools.

## Configuration

The RAG system is controlled in `mod_ollama_chat.conf`:
- `OllamaChat.EnableRAG`
- `OllamaChat.RAGDataPath`
- `OllamaChat.RAGSimilarityThreshold` (default 0.3)
- `OllamaChat.RAGMaxRetrievedItems` (default 3)
- `OllamaChat.RAGPromptTemplate`

When `OllamaChat.ChatPromptTemplate` is empty, the optional prompt composer also supports `OllamaChat.RAGKnowledgeChance` and related zone/home/level modifiers.

## Examples

### Query Matching
If a player asks "How do I get better at blacksmithing?", the system might retrieve entries about:
- Blacksmithing profession details
- Required materials and leveling
- Specializations and recipes
- Related quests and trainers

### Response Enhancement
The bot's response would be enhanced with relevant information like:
- Blacksmithing trainers locations
- Key materials needed
- Profitability information
- Related achievements

This system allows the bot to provide helpful, contextual WoW information without requiring hardcoded responses for every possible question.