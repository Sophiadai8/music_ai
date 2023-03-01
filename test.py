from pythonosc import osc_message_builder
from pythonosc import udp_client
import time
sender = udp_client.SimpleUDPClient('127.0.0.1', 4560)

print("What note would you like to play? (type 'stop' to stop)")

while (True):
    note = input().title()
    if note == "stop":
        break
    sender.send_message('/send', [note, 0.6, 2, "piano"])
    time.sleep(0.6)
print("Thank you for playing")

# Sonic Pi:

# live_loop :send do
#   use_real_time
#   a, b, c, d = sync "/osc*/send"
#   use_synth d
#   play a, sustain: b, amp: c
# end