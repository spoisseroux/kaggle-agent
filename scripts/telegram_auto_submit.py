#!/usr/bin/env python3
"""
Auto-submit daemon that monitors message_bus and automatically
presses Enter in the tmux Claude Code session to submit messages.

This uses the user's Claude subscription (no API costs) while
providing automatic message submission.
"""
import os
import sys
import time
import sqlite3
import subprocess
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path.home() / "kaggle-agent/message_bus.sqlite3"
POLL_INTERVAL = 5  # seconds - check frequently for new messages
TMUX_SESSION = "kaggle"  # Default tmux session name


def get_new_message_count():
    """Count unprocessed messages"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM messages
        WHERE role = 'human' AND status = 'new'
    """)

    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_new_messages():
    """Get unprocessed messages with text"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, text
        FROM messages
        WHERE role = 'human' AND status = 'new'
        ORDER BY created_at ASC
    """)

    messages = cursor.fetchall()
    conn.close()
    return messages


def is_session_attached(session_name):
    """Check if a tmux session is currently attached"""
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name} #{session_attached}"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return False

        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == session_name:
                return parts[1] == "1"

    except Exception as e:
        print(f"Error checking session attachment: {e}")

    return False


def find_claude_pane():
    """Find the tmux pane running Claude Code"""
    try:
        result = subprocess.run(
            ["tmux", "list-panes", "-a", "-F", "#{session_name}:#{window_index}.#{pane_index} #{pane_current_command} #{pane_pid}"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return None

        # Look for claude process
        for line in result.stdout.splitlines():
            if "claude" in line.lower():
                pane = line.split()[0]  # session:window.pane
                return pane

    except Exception as e:
        print(f"Error finding Claude pane: {e}")

    return None


def send_enter_to_pane(pane):
    """Send Enter key to the specified tmux pane"""
    try:
        # Send Enter key
        result = subprocess.run(
            ["tmux", "send-keys", "-t", pane, "Enter"],
            capture_output=True,
            timeout=5
        )

        return result.returncode == 0

    except Exception as e:
        print(f"Error sending Enter: {e}")
        return False


def mark_messages_as_pending():
    """
    Mark new messages as pending (they'll be processed once Enter is pressed)
    This prevents double-counting
    """
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE messages
        SET status = 'pending', updated_at = ?
        WHERE role = 'human' AND status = 'new'
    """, (time.time(),))

    count = cursor.rowcount
    conn.commit()
    conn.close()

    return count


def main():
    """Main daemon loop"""
    print("🤖 Telegram Auto-Submit Daemon Started")
    print(f"   Monitoring: {DB_PATH}")
    print(f"   Poll interval: {POLL_INTERVAL}s")

    last_processed_ids = set()
    claude_pane = None
    last_pane_check = 0

    while True:
        try:
            # Re-find Claude pane every 60 seconds in case it changes
            if time.time() - last_pane_check > 60:
                claude_pane = find_claude_pane()
                last_pane_check = time.time()
                if claude_pane:
                    print(f"   Found Claude in pane: {claude_pane}")
                else:
                    print("   ⚠️ Claude pane not found - will retry")

            # Check for new messages
            new_messages = get_new_messages()

            # Filter to only messages we haven't seen before
            unprocessed = [msg for msg in new_messages if msg[0] not in last_processed_ids]

            if unprocessed:
                print(f"\n📨 {len(unprocessed)} new message(s) detected")

                # Calculate delay based on message length
                # Long pastes need more time for Claude Code to process
                total_length = sum(len(msg[1]) for msg in unprocessed)
                if total_length > 1000:
                    delay = 10  # 10 seconds for long messages - Claude needs time
                    print(f"   Long message detected ({total_length} chars) - waiting {delay}s")
                else:
                    delay = 2  # 2 seconds for short messages (was 1, but be safer)

                # Wait for Claude Code to pre-fill the input
                time.sleep(delay)

                if claude_pane:
                    print(f"   Sending Enter to {claude_pane}...")

                    # Send Enter key - just once, but after long delay
                    success = send_enter_to_pane(claude_pane)
                    if success:
                        print(f"   ✅ Enter sent successfully")
                        # Mark messages as pending so we don't re-process
                        marked = mark_messages_as_pending()
                        print(f"   Marked {marked} messages as pending")

                        # Remember we processed these messages
                        for msg in unprocessed:
                            last_processed_ids.add(msg[0])
                    else:
                        print(f"   ❌ Failed to send Enter")
                else:
                    print(f"   ❌ No Claude pane found - can't auto-submit")
                    print(f"       Please start Claude Code in tmux session")

            # Sleep before next poll
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\n🛑 Shutting down auto-submit daemon")
            break
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)  # Wait longer on error


if __name__ == "__main__":
    # Check if tmux is available
    try:
        subprocess.run(["tmux", "-V"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ tmux not found! This script requires tmux.")
        sys.exit(1)

    main()
