# Lucky Train AI System

This document outlines the various AI model types and database connectors available in the Lucky Train AI assistant system.

## AI Model Types

The system supports multiple variations of AI models:

### 1. Narrow AI (ANI)
- **Type:** `ani`
- **Description:** Specialized in the Lucky Train domain, providing focused responses about the project.
- **Use Case:** Answering specific questions about the Lucky Train project, token, and blockchain.

### 2. General AI (AGI)
- **Type:** `agi`
- **Description:** More versatile AI with cross-domain knowledge and reasoning capabilities.
- **Use Case:** Handling questions that require understanding multiple domains and concepts.

### 3. Super Intelligence (ASI)
- **Type:** `asi`
- **Description:** Advanced AI with superior reasoning capabilities, using GPT-4o for complex analysis.
- **Use Case:** Deep analysis, prediction, and handling multifaceted problems.

### 4. Machine Learning
- **Type:** `machine_learning`
- **Description:** Uses classic ML algorithms like TF-IDF for pattern recognition.
- **Use Case:** Statistical analysis of text and simple pattern matching.

### 5. Deep Learning
- **Type:** `deep_learning`
- **Description:** Uses neural networks for feature extraction and representation learning.
- **Use Case:** Understanding complex patterns and semantic meaning.

### 6. Reinforcement Learning
- **Type:** `reinforcement_learning`
- **Description:** Model that improves through trial and error and reward-based feedback.
- **Use Case:** Adapting to user preferences and learning from interactions.

### 7. Analytical AI
- **Type:** `analytical_ai`
- **Description:** Focused on data analysis, trends, and insights.
- **Use Case:** Analyzing project metrics, token performance, or user statistics.

### 8. Interactive AI
- **Type:** `interactive_ai`
- **Description:** Optimized for natural conversation and engagement.
- **Use Case:** Dynamic chat interactions with better memory of conversation history.

### 9. Functional AI
- **Type:** `functional_ai`
- **Description:** Performs specific tasks and actions like retrieving token prices.
- **Use Case:** Executing operations and providing action-oriented responses.

### 10. Symbolic Systems
- **Type:** `symbolic_systems`
- **Description:** Rule-based logical reasoning system using explicit knowledge.
- **Use Case:** Questions requiring logical deduction and explicit reasoning.

### 11. Connectionist Systems
- **Type:** `connectionist_systems`
- **Description:** Neural network-based system focused on patterns and associations.
- **Use Case:** Recognition of patterns and generalization from examples.

### 12. Hybrid Systems
- **Type:** `hybrid_systems`
- **Description:** Combines symbolic and connectionist approaches for balanced reasoning.
- **Use Case:** Complex problems requiring both logical rules and pattern recognition.

## Database Connectors

The system supports connecting to various databases:

### 1. Chat2DB
- **Type:** `chat2db`
- **Description:** SQL database for chat logs and user interactions.

### 2. Google BigQuery
- **Type:** `bigquery`
- **Description:** For large-scale data analytics and processing.

### 3. Amazon Aurora
- **Type:** `aurora`
- **Description:** Relational database for structured data.

### 4. Microsoft Azure Cosmos DB
- **Type:** `cosmosdb`
- **Description:** NoSQL database for flexible data models.

### 5. Snowflake
- **Type:** `snowflake`
- **Description:** Data warehouse for analytics and big data.

### 6. IBM Db2 AI
- **Type:** `db2ai`
- **Description:** AI-enhanced database for advanced analytics.

## Usage

### Command Line Options

When running the assistant, you can specify the AI model type:

```bash
python src/main.py console --ai-model agi
```

Available options for `--ai-model`:
- `ani`, `agi`, `asi`
- `machine_learning`, `deep_learning`, `reinforcement_learning`
- `analytical_ai`, `interactive_ai`, `functional_ai`
- `symbolic_systems`, `connectionist_systems`, `hybrid_systems`

You can also specify a database connector:

```bash
python src/main.py console --db-connector bigquery
```

### Console Commands

When running in console mode, you can use these commands:

- `/exit` - Exit the console
- `/models` - List available AI models
- `/model <name>` - Switch to the specified AI model type
- `/dbs` - List available database connectors
- `/db <name> <query>` - Execute a query on the specified database

Example:
```
> /model agi
Switched to model: agi

> /db bigquery SELECT * FROM users LIMIT 5
```

### Configuration

You can also configure the default AI model and enabled models in the configuration file:

```json
{
  "default_ai_model": "ani",
  "current_ai_model": "ani",
  "enabled_ai_models": [
    "ani", 
    "agi", 
    "machine_learning"
  ]
}
```

## Environment Variables

Set these environment variables to enable database connectors:

- `CHAT2DB_API_ENDPOINT` - Endpoint for Chat2DB
- `BIGQUERY_PROJECT_ID` - Google Cloud project ID
- `AURORA_HOST` - Amazon Aurora database host
- `COSMOS_ENDPOINT` - Azure Cosmos DB endpoint
- `SNOWFLAKE_ACCOUNT` - Snowflake account identifier

## Extending the System

To add new AI model types or database connectors:

1. Create a new class inheriting from `BaseAIModel` or `BaseDBConnector`
2. Implement the required methods
3. Add the new class to the appropriate factory function
4. Update the configuration file to enable the new component 