tmux new-session -d -s dreamer_mw -n h15

cd ~/579/event-horizon

tmux send-keys -t dreamer_mw:h15 "bash scripts/train_dreamer_mw.sh metaworld_pick-place 0 15 501000 2" Enter

tmux attach -t dreamer_mw