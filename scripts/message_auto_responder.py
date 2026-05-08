#!/usr/bin/env python3
"""
Auto-responder daemon that monitors message_bus and responds to Telegram messages
using the Anthropic API directly, bypassing Claude Code UI.
"""
import os
import sys
import time
import sqlite3
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from core.notify import send_notification

DB_PATH = Path.home() / "kaggle-agent/message_bus.sqlite3"
POLL_INTERVAL = 10  # seconds
MAX_CONTEXT_MESSAGES = 20  # Keep last N messages for context


def get_new_messages():
    """Get unprocessed messages from message_bus"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, text, source, created_at
        FROM messages
        WHERE role = 'human' AND status = 'new'
        ORDER BY created_at ASC
    """)

    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return messages


def mark_message_processed(msg_id):
    """Mark a message as claimed/processed"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE messages
        SET status = 'claimed', updated_at = ?
        WHERE id = ?
    """, (time.time(), msg_id))

    conn.commit()
    conn.close()


def get_recent_context():
    """Get recent conversation for context"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT role, text, source
        FROM messages
        WHERE status IN ('claimed', 'consumed', 'delivered')
        ORDER BY created_at DESC
        LIMIT ?
    """, (MAX_CONTEXT_MESSAGES,))

    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return list(reversed(messages))  # Chronological order


def respond_to_message(message_text, context):
    """Use Anthropic API to generate response"""
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Build conversation history
    conversation = []
    for ctx_msg in context[-10:]:  # Last 10 for context
        role = "user" if ctx_msg["role"] == "human" else "assistant"
        conversation.append({
            "role": role,
            "content": ctx_msg["text"]
        })

    # Add current message
    conversation.append({
        "role": "user",
        "content": message_text
    })

    # System prompt
    system_prompt = """You are the Kaggle competition agent. You're currently working on the
store-sales-time-series-forecasting competition.

Best model so far: XGBoost v1 (LB 0.526)
Target: ~0.377 (top leaderboard)
Status: Hit daily submission limit (5/5), session paused

Be concise and format for mobile Telegram:
- Short lines
- Clear breaks
- Bullet points with (-)
- Emoji headers
- Max 10 lines per message

You have access to the full kaggle-agent codebase and can execute commands."""

    # Call API
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=conversation
    )

    return response.content[0].text


def save_agent_response(response_text):
    """Save agent's response to message_bus"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    import uuid
    msg_id = str(uuid.uuid4()).replace('-', '')
    now = time.time()

    cursor.execute("""
        INSERT INTO messages (id, role, source, text, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (msg_id, 'agent', 'auto_responder', response_text, 'delivered', now, now))

    conn.commit()
    conn.close()


def main():
    """Main daemon loop"""
    print(f"🤖 Auto-responder daemon started")
    print(f"   Polling {DB_PATH} every {POLL_INTERVAL}s")
    print(f"   Using API key: {os.environ.get('ANTHROPIC_API_KEY', 'NOT SET')[:20]}...")

    while True:
        try:
            # Check for new messages
            new_msgs = get_new_messages()

            if new_msgs:
                print(f"\n📨 Found {len(new_msgs)} new message(s)")

                # Get context
                context = get_recent_context()

                for msg in new_msgs:
                    print(f"   Processing: {msg['text'][:60]}...")

                    # Generate response
                    try:
                        response = respond_to_message(msg['text'], context)

                        # Send to Telegram
                        send_notification(response)

                        # Save to message_bus
                        save_agent_response(response)

                        # Mark as processed
                        mark_message_processed(msg['id'])

                        print(f"   ✅ Responded and sent to Telegram")

                        # Add to context for next message
                        context.append({
                            'role': 'human',
                            'text': msg['text'],
                            'source': msg['source']
                        })
                        context.append({
                            'role': 'agent',
                            'text': response,
                            'source': 'auto_responder'
                        })

                    except Exception as e:
                        print(f"   ❌ Error responding: {e}")
                        send_notification(f"⚠️ Auto-responder error: {str(e)[:100]}")

            # Sleep before next poll
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\n🛑 Shutting down auto-responder")
            break
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            time.sleep(60)  # Wait longer on error


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set!")
        sys.exit(1)

    main()
