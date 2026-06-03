tmux new-session -d -s tdmpc2_mw -n h3
tmux new-window -t tdmpc2_mw -n h5
tmux new-window -t tdmpc2_mw -n h10
tmux new-window -t tdmpc2_mw -n h15

cd ~/579/event-horizon

tmux send-keys -t tdmpc2_mw:h3 "bash scripts/train_tdmpc2_mw.sh mw-pick-place 1 3 500000 4" Enter
tmux send-keys -t tdmpc2_mw:h5 "bash scripts/train_tdmpc2_mw.sh mw-pick-place 1 5 500000 5" Enter
tmux send-keys -t tdmpc2_mw:h10 "bash scripts/train_tdmpc2_mw.sh mw-pick-place 1 10 500000 6" Enter
tmux send-keys -t tdmpc2_mw:h15 "bash scripts/train_tdmpc2_mw.sh mw-pick-place 1 15 500000 7" Enter

tmux attach -t tdmpc2_mw