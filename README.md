# Public Service Navigation Assistant

A voice-enabled AI system designed to help citizens access and understand support programs such as SNAP, housing assistance, and healthcare benefits. Built using fully open-source technologies, it combines conversational AI, Retrieval-Augmented Generation (RAG), and speech capabilities to provide real-time, accurate, and accessible answers through a natural voice interface.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Voice Input   │───▶│  Speech-to-Text │───▶│  NLU + Dialog   │
│   (Twilio)      │    │   (Whisper)     │    │   (Rasa)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Voice Output   │◀───│  TTS + SSML     │◀───│  LLM + RAG      │
│   (Twilio)      │    │  (gTTS/Coqui)   │    │  (LangChain)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Features

- **Voice Interface**: Natural phone-based interaction using Twilio
- **Web Chat Interface**: Modern web-based chat for text interactions
- **Speech Recognition**: Whisper-based speech-to-text conversion
- **Conversational AI**: Rasa-powered intent recognition and dialog management
- **Knowledge Retrieval**: RAG pipeline with vector database for document search
- **LLM Integration**: Local LLM (Ollama) for response generation
- **Agentic AI Orchestration**: Planner + tool execution + session memory for multi-step reasoning
- **Speech Synthesis**: Local TTS with SSML for natural speech
- **Fully Open Source**: No proprietary dependencies (except optional Twilio trial credits)

## Agentic AI Upgrade

The backend now supports two execution modes:

- **classic**: Existing RAG -> LLM pipeline (default, backward-compatible)
- **agentic**: Multi-step planning loop with retrieval tool calls and session memory

Agentic mode enables:

- Session-based multi-turn context with `session_id`
- Structured execution trace (`agent_trace`) for observability
- Optional dedicated endpoint for agent-only workflows

### Agentic API Examples

Use the existing `/query` endpoint with `mode=agentic`:

```bash
curl -X POST http://localhost:8000/query \
   -H "Content-Type: application/json" \
   -d '{
      "query": "I am a single parent. What benefits can I apply for first?",
      "mode": "agentic",
      "session_id": "citizen-001",
      "max_steps": 4
   }'
```

Or use the dedicated endpoint:

```bash
curl -X POST http://localhost:8000/agent/query \
   -H "Content-Type: application/json" \
   -d '{
      "query": "What documents should I prepare for SNAP and housing assistance?",
      "session_id": "citizen-001"
   }'
```

Inspect and clear sessions:

```bash
curl http://localhost:8000/agent/sessions/citizen-001
curl -X DELETE http://localhost:8000/agent/sessions/citizen-001
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Twilio account (optional - for voice calls)
- OpenAI API key (optional - for Whisper fallback)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd public-service-navigation
   ```

2. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

4. **Train Rasa model**
   ```bash
   cd rasa
   rasa train
   ```

5. **Access the interfaces**
   - Web Chat: Open `http://localhost:8080/chat_interface.html`
   - API Documentation: Open `http://localhost:8000/docs`

## 📁 Project Structure

```
public-service-navigation/
├── backend/                 # FastAPI RAG backend
│   ├── app/
│   │   ├── models/         # Pydantic models
│   │   ├── services/       # RAG and LLM services
│   │   └── api/           # API endpoints
│   ├── data/              # Knowledge base documents
│   └── main.py           # FastAPI application
├── rasa/                  # Rasa conversational AI
│   ├── data/             # Training data
│   ├── actions/          # Custom actions
│   └── config.yml        # Rasa configuration
├── voice/                # Voice processing utilities
│   ├── speech_recognition/
│   ├── speech_synthesis/
│   └── twilio_integration/
├── chat_interface.html   # Web chat interface
├── docker-compose.yml    # Service orchestration
└── terraform/           # Infrastructure as code
```

## Configuration

### Environment Variables

```bash
# Twilio Configuration (Optional - for voice features)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_twilio_number

# OpenAI Configuration (Optional - fallback for Whisper)
OPENAI_API_KEY=your_openai_key

# LLM Configuration
LLM_MODEL=llama2
LLM_API_BASE=http://localhost:11434  # Ollama endpoint

# Agentic Orchestration Configuration
AGENT_MAX_STEPS=4
AGENT_TRACE_ENABLED=true

# Service URLs
RASA_WEBHOOK_URL=http://localhost:5005/webhooks/rest/webhook
RAG_BACKEND_URL=http://localhost:8000
```

## Use Cases

The assistant helps users with:

- **SNAP Benefits**: Eligibility, application process, benefit amounts
- **Housing Assistance**: Section 8, public housing, emergency shelter
- **Healthcare**: Medicaid, Medicare, ACA marketplace
- **General Navigation**: Finding local offices, required documents

## Testing

Run the test suite to verify all components are working:

```bash
python test_system.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Rasa for conversational AI framework
- LangChain for RAG pipeline
- Twilio for voice communication
- OpenAI for speech recognition
- The open-source community for LLM models 
