tmux new-session -d -s dreamer_mw_h5 -n h5

cd ~/579/event-horizon

tmux send-keys -t dreamer_mw_h5:h5 "bash scripts/train_dreamer_mw.sh metaworld_pick-place 0 5 501000 7" Enter

tmux attach -t dreamer_mw_h5