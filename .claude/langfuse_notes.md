# Langfuse Integration Notes

## Setup Complete ✅

- **Server**: Langfuse 2.0 running on homelab (http://docker:3000)
- **Client**: Langfuse Python SDK 4.6.1
- **Project URL**: http://docker:3000/project/cmoxvzksu0006rttfvmavgrc9
- **Status**: Working with minor warnings

## Known Issues

### 404 Warning During Export
**Error**: `Failed to export span batch code: 404, reason: Not Found`

**Cause**: Version mismatch between client SDK (4.6.1) and server (2.0). The client is trying to use a newer API endpoint that doesn't exist in the server.

**Impact**: Traces are still created and visible in the UI. The 404 is for telemetry/metrics export, not the actual trace data.

**Resolution Options**:
1. **Ignore**: Traces work fine, just noisy logs
2. **Upgrade server**: Deploy Langfuse 3.x (might require migrations)
3. **Downgrade client**: Use older Python SDK (not recommended)

**Recommendation**: Ignore for now. Upgrade server when convenient.

## API Usage

### Correct SDK v4+ Pattern

```python
from core.langfuse_integration import get_langfuse_client, trace_llm_call

# Automatic tracing (integrated into llm_interface.py)
from core.llm_interface import ask_ollama
response = ask_ollama("prompt", agent="Developer")
# Automatically traced to Langfuse!

# Manual tracing
trace_llm_call(
    func_name="my_function",
    model="qwen3:14b",
    prompt="input text",
    response="output text",
    duration=1.5,
    agent="MyAgent",
    metadata={"custom": "data"}
)

# Always flush in short-lived scripts
client = get_langfuse_client()
client.flush()
```

### Context Manager Pattern (Lower Level)

```python
client = get_langfuse_client()

with client.start_as_current_observation(
    as_type="generation",
    name="my-llm-call",
    model="gpt-4",
) as gen:
    gen.update(
        input="prompt",
        output="response",
        usage={"input": 10, "output": 20}
    )
```

## Integration Points

### LLM Interface
All calls through `core.llm_interface.py` are automatically traced:
- `ask_ollama()` → traced with model, agent, think mode
- `ask_claude()` → traced with model, agent, escalation reason

### Agents
All agent LLM calls are automatically tagged:
- Reader → "Reader/ollama"
- Planner → "Planner/ollama"
- Developer → "Developer/ollama" or "Developer/claude"
- Research → "Research/claude"
- Reviewer → "Reviewer/ollama"

### Cost Tracking
- Claude Sonnet 4.5: $3/M input, $15/M output
- Ollama: $0 (local)
- Costs calculated and logged per trace

## Viewing Traces

1. Open Langfuse UI: http://docker:3000
2. Go to project: kaggle-agent
3. Navigate to Traces tab
4. Filter by:
   - Agent name (metadata)
   - Model
   - Time range
   - Cost

## Frontend Integration

Langfuse link added to `/health` endpoint in `api/main.py`:

```json
{
  "services": {
    ...
    "langfuse": {
      "url": "http://docker:3000/project/cmoxvzksu0006rttfvmavgrc9",
      "type": "external_link",
      "status": true/false
    }
  }
}
```

Frontend should render this as a clickable link/button.

## Future Improvements

1. **Upgrade Langfuse server** to 3.x to match client SDK
2. **Add custom dashboards** in Langfuse UI
3. **Cost alerts** when spending exceeds threshold
4. **Performance monitoring** for slow LLM calls
5. **A/B testing** different prompts/models

## Resources

- [Langfuse Python SDK Docs](https://langfuse.com/docs/sdk/python)
- [Decorator-Based Integration](https://langfuse.com/docs/sdk/python/decorators)
- [Advanced Usage](https://langfuse.com/docs/observability/sdk/python/advanced-usage)
