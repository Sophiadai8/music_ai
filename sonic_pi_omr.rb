live_loop :right do
  use_real_time
  a, b, c, d = sync "/osc*/sci/thing"
  use_synth d
  play a, sustain: b, amp: c
end