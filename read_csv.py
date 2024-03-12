import csv

with open('/amax/home/xtyao/cir/MMPT/models/combiner_trained_on_fiq_ViT-B/16_2023-12-27_01:35:11/validation_metrics.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    res = []
    # for row in csv_reader:
    #     if row[0] != 'epoch':
    #         res.append(row[-1])
    # print(max(res))
    for row in csv_reader:
        if row[0] != 'epoch':
            res.append(row)
    if len(res[0]) == 10:
        res.sort(key=lambda x: x[-1])

        print(f'dress R@10 = {float(res[-1][1]):.2f}')
        print(f'dress R@50 = {float(res[-1][2]):.2f}')

        print(f'shirt R@10 = {float(res[-1][5]):.2f}')
        print(f'shirt R@50 = {float(res[-1][6]):.2f}')

        print(f'toptee R@10 = {float(res[-1][3]):.2f}')
        print(f'toptee R@50 = {float(res[-1][4]):.2f}')
        
        print(f'average R@10 = {float(res[-1][7]):.2f}')
        print(f'average R@50 = {float(res[-1][8]):.2f}')
        print(f'Avg. Recall = {float(res[-1][9]):.2f}')
    else:
        res.sort(key=lambda x: x[-4])
        
        print(f"recall_at1 = {float(res[-1][4]):.2f}")
        print(f"recall_at5 = {float(res[-1][5]):.2f}")
        print(f"recall_at10 = {float(res[-1][6]):.2f}")
        print(f"recall_at50 = {float(res[-1][7]):.2f}")

        print(f"group_recall_at1 = {float(res[-1][1]):.2f}")
        print(f"group_recall_at2 = {float(res[-1][2]):.2f}")
        print(f"group_recall_at3 = {float(res[-1][3]):.2f}")

        print(f"mean(R@5+R_s@1) = {float(res[-1][8]):.2f}")

        print(f"\narithmetic_mean = {float(res[-1][9]):.2f}")
        print(f"harmonic_mean = {float(res[-1][10]):.2f}")
        print(f"geometric_mean = {float(res[-1][11]):.2f}")





