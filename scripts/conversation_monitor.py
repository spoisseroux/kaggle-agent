#!/usr/bin/env python3
"""Conversation Monitor - Auto-forward assistant responses to Telegram

Monitors Claude Code conversation transcript in real-time and automatically
forwards assistant messages to Telegram if they weren't already sent via notify.py.

Hybrid approach:
1. Tails conversation log (like a background service)
2. Detects assistant messages
3. Checks if already sent to Telegram (by looking for recent notify.py calls)
4. Auto-forwards if missing

Run as systemd service for 24/7 operation.
"""
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

# Find the active conversation transcript
CLAUDE_DIR = Path.home() / ".claude" / "projects" / "-home-keehar-kaggle-agent"

def find_active_conversation():
    """Find the most recently modified conversation file."""
    conversations = list(CLAUDE_DIR.glob("*.jsonl"))
    if not conversations:
        return None
    return max(conversations, key=lambda p: p.stat().st_mtime)

def tail_conversation(filepath):
    """Tail a conversation file and yield new messages."""
    # Start from end of file
    with open(filepath, 'r') as f:
        f.seek(0, 2)  # Go to end
        while True:
            line = f.readline()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
            else:
                time.sleep(0.5)  # Wait for new content

def is_assistant_message(msg):
    """Check if message is from assistant."""
    return msg.get("role") == "assistant" and msg.get("type") == "message"

def extract_text_content(msg):
    """Extract text content from assistant message."""
    content = msg.get("content", [])
    if isinstance(content, str):
        return content

    # Content is array of content blocks
    text_parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))

    return "\n\n".join(text_parts).strip()

def should_forward(text, recent_notifies):
    """Check if message should be forwarded to Telegram."""
    if not text or len(text) < 20:
        # Skip very short messages (likely just acknowledgments)
        return False

    # Check if this exact text was recently sent via notify.py
    # (We track recent notify calls to avoid duplicates)
    for notify_text in recent_notifies:
        if notify_text in text or text in notify_text:
            return False

    return True

def send_to_telegram(text, max_length=3800):
    """Send message to Telegram via notify.py."""
    # Telegram has ~4096 char limit, we use 3800 to be safe
    if len(text) > max_length:
        # Split into chunks
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        for i, chunk in enumerate(chunks):
            prefix = f"[Part {i+1}/{len(chunks)}]\n\n" if len(chunks) > 1 else ""
            subprocess.run(
                ["python", "core/notify.py", prefix + chunk],
                cwd="/home/keehar/kaggle-agent",
                capture_output=True
            )
            time.sleep(1)  # Rate limit
    else:
        subprocess.run(
            ["python", "core/notify.py", text],
            cwd="/home/keehar/kaggle-agent",
            capture_output=True
        )

def monitor_bash_outputs():
    """Monitor post_bash.py hook to track notify.py calls."""
    # This would parse recent bash commands to see if notify.py was called
    # For now, we'll use a simpler heuristic: recent_notifies list
    return []

def main():
    print("🔍 Conversation Monitor starting...")

    recent_notifies = []  # Track recent notify.py calls to avoid duplicates
    last_cleanup = datetime.now()

    while True:
        try:
            conv_file = find_active_conversation()
            if not conv_file:
                print("⚠️  No active conversation found, waiting...")
                time.sleep(10)
                continue

            print(f"📝 Monitoring: {conv_file.name}")

            for msg in tail_conversation(conv_file):
                # Check if conversation file changed (new session)
                current_active = find_active_conversation()
                if current_active != conv_file:
                    print(f"🔄 Conversation switched to: {current_active.name}")
                    break

                # Process assistant messages
                if is_assistant_message(msg):
                    text = extract_text_content(msg)

                    if should_forward(text, recent_notifies):
                        print(f"📤 Auto-forwarding to Telegram ({len(text)} chars)")
                        send_to_telegram(text)

                        # Track this message to avoid re-sending
                        recent_notifies.append(text[:200])  # Store first 200 chars as fingerprint

                # Cleanup old fingerprints every 5 minutes
                if datetime.now() - last_cleanup > timedelta(minutes=5):
                    recent_notifies = recent_notifies[-10:]  # Keep last 10
                    last_cleanup = datetime.now()

        except KeyboardInterrupt:
            print("\n👋 Monitor stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(5)  # Wait before retry

if __name__ == "__main__":
    main()
