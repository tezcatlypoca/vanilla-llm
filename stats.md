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

    multi-gpu - batch 500
    - 10 epocs -> 11,53% (389)
    - 20 epocs -> 12,30% (773,3) 
    - 30 epocs -> % (1154)

