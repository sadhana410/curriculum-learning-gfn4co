CUDA_VISIBLE_DEVICES=0 python problems/graph_coloring/main.py --conditional --graphs myciel2 --max-colors 191 --steps 10000 &
CUDA_VISIBLE_DEVICES=1 python problems/graph_coloring/main.py --conditional --graphs myciel2 myciel3 --max-colors 191 --steps 10000 &
CUDA_VISIBLE_DEVICES=2 python problems/graph_coloring/main.py --conditional --graphs myciel2 myciel3 myciel4 --max-colors 191 --steps 10000 &
CUDA_VISIBLE_DEVICES=3 python problems/graph_coloring/main.py --conditional --graphs myciel2 myciel3 myciel4 myciel5 --max-colors 191 --steps 10000 &