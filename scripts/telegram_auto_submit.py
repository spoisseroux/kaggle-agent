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

    last_msg_count = 0
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
            msg_count = get_new_message_count()

            if msg_count > 0 and msg_count != last_msg_count:
                print(f"\n📨 {msg_count} new message(s) detected")

                # Wait a moment for Claude Code to pre-fill the input
                time.sleep(1)

                if claude_pane:
                    print(f"   Sending Enter to {claude_pane}...")

                    # Send Enter key
                    if send_enter_to_pane(claude_pane):
                        print(f"   ✅ Enter sent successfully")

                        # Mark messages as pending so we don't re-process
                        marked = mark_messages_as_pending()
                        print(f"   Marked {marked} messages as pending")
                    else:
                        print(f"   ❌ Failed to send Enter")
                else:
                    print(f"   ❌ No Claude pane found - can't auto-submit")
                    print(f"       Please start Claude Code in tmux session")

                last_msg_count = msg_count

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
