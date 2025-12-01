CUDA_VISIBLE_DEVICES=0 python problems/graph_coloring/evaluate.py --conditional --checkpoint problems/graph_coloring/checkpoints/conditional_myciel2_K191_final.pt --graphs myciel7
CUDA_VISIBLE_DEVICES=1 python problems/graph_coloring/evaluate.py --conditional --checkpoint problems/graph_coloring/checkpoints/conditional_myciel2-myciel3_K191_final.pt --graphs myciel7
CUDA_VISIBLE_DEVICES=2 python problems/graph_coloring/evaluate.py --conditional --checkpoint problems/graph_coloring/checkpoints/conditional_myciel2-myciel3-myciel4_K191_final.pt --graphs myciel7
CUDA_VISIBLE_DEVICES=3 python problems/graph_coloring/evaluate.py --conditional --checkpoint problems/graph_coloring/checkpoints/conditional_myciel2-myciel3-myciel4-myciel5_K191_final.pt --graphs myciel7
