# OpenAI API Integration - Lucky Train AI Assistant

## Overview

This document explains how OpenAI API was integrated into the Lucky Train AI Assistant project. The integration allows the assistant to use powerful GPT models for natural language understanding and generation.

## Configuration Details

The following environment variables need to be configured in your `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORGANIZATION_ID=your_organization_id_here
DEFAULT_AI_MODEL=gpt-4o-mini
```

## Files Updated

The following files have been modified to include the OpenAI API configuration:

1. `.env` - Environment variables file with API keys
2. `src/ai_models.py` - Core AI models definition including OpenAI-based models

## Test Scripts

Two test scripts have been created to verify the OpenAI API integration:

1. `openai_config_test.py` - Python script to test the OpenAI connection and update configuration files
2. `openai_config.js` - JavaScript module to test the OpenAI connection from Node.js

## How to Test the Integration

### Python

Run the Python test script:

```bash
python openai_config_test.py
```

This script will:
- Test the connection to OpenAI API using environment variables
- Display a sample response

### JavaScript (Node.js)

To test the JavaScript integration (requires installing the OpenAI npm package):

```bash
npm install openai
node openai_config.js
```

## Available AI Models

The following AI models can be used with the OpenAI integration:

1. **NarrowAI (ANI)** - Domain-specific AI
2. **GeneralAI (AGI)** - Cross-domain knowledge using OpenAI models
3. **SuperIntelligence (ASI)** - Advanced capabilities using OpenAI's best models
4. **MachineLearning** - Classical ML algorithms
5. **DeepLearning** - Neural network-based models
6. **ReinforcementLearning** - Adaptive learning models
7. **AnalyticalAI** - Data analysis focused
8. **InteractiveAI** - Conversation-based models
9. **FunctionalAI** - Task-oriented models
10. **SymbolicSystems** - Logic and rule-based models
11. **ConnectionistSystems** - Neural network approaches
12. **HybridSystems** - Combined approach models

## Usage Examples

### Python Example

```python
from src.ai_models import create_ai_model

# Create an OpenAI-powered AGI model
agi_model = create_ai_model("agi", {"model": "gpt-4o-mini"})

# Generate a response
response = agi_model.generate_response("Как использовать криптовалюту в проекте?")
print(response["response"])
```

### JavaScript Example

```javascript
const { openai } = require('./openai_config');

async function askQuestion(question) {
  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      { role: 'system', content: 'Ты ассистент проекта Lucky Train. Отвечай на украинском.' },
      { role: 'user', content: question }
    ],
    temperature: 0.7,
  });
  
  return response.choices[0].message.content;
}

// Example usage
askQuestion('Как использовать Lucky Train?').then(console.log);
```

## Security Notes

- Never commit API keys to version control
- Use environment variables or `.env` files for local development
- Use GitHub Secrets for CI/CD processes
- Keep your API keys secure and never expose them in client-side code
- Consider using a proxy service or rate limiting to control API costs 