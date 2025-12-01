13M Params:
    On CPU computing(static LR):
    - 10 epocs -> 5% precision
    - 50 epocs -> 11,35% precision

    on CPU computing (dynamic LR):
    - 10 epocs -> 12,33% (exec time: 525s)
    - 20 epocs -> 15,44% (995s)
    - 30 epocs -> 16,99% (1534s)

235M Params:
    on multi-GPU:
    - 10 epocs -> 19,39% (411s) 1er exec sous multi gpu
    - 20 epocs -> 11,37% (810s)
    - 30 epocs -> 16,73% (1220s)

    multi -gpu - batch 1000
    - 10 epocs -> 11,35% (385,9)
    - 20 epocs -> 16,55% (769,2) 
    - 30 epocs -> 12,32% (1154)
    - 100 epocs -> 14,48% (3809)

    multi-gpu - batch 500
    - 10 epocs -> 11,53% (389)
    - 20 epocs -> 12,30% (773,3) 
    - 30 epocs -> 13,84% (1154)
    - 100 epocs -> 12,93% (3870)

==========================================================================

Nouvelle archi
235 param - multi gpu thread

10 epochs -> 84% (42s) - 83.5% - 85,11%
30 epochs -> 85,7% - 87,6% 
100 epochs -> 90,53%