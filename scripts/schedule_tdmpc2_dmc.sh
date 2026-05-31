tmux new-session -d -s tdmpc2 -n h3
tmux new-window -t tdmpc2 -n h5
tmux new-window -t tdmpc2 -n h10
tmux new-window -t tdmpc2 -n h15

cd ~/579/event-horizon

tmux send-keys -t tdmpc2:h3 "bash scripts/train_tdmpc2_dmc.sh dmc_walker_walk 1 3 200000 4" Enter
tmux send-keys -t tdmpc2:h5 "bash scripts/train_tdmpc2_dmc.sh dmc_walker_walk 1 5 200000 5" Enter
tmux send-keys -t tdmpc2:h10 "bash scripts/train_tdmpc2_dmc.sh dmc_walker_walk 1 10 200000 6" Enter
tmux send-keys -t tdmpc2:h15 "bash scripts/train_tdmpc2_dmc.sh dmc_walker_walk 1 15 200000 7" Enter

tmux attach -t tdmpc2