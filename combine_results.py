import json
with open(f'demo_out/results_dino_davis.json','r') as f:
    dino_results_dict = json.load(f)
with open(f'demo_out/results_kalman_davis_constraint.json','r') as f:
    kalman_results_dict = json.load(f)
    
total = 0
OA = 0
AJ = 0
for key in dino_results_dict.keys():
    if key == "average":
        continue
    dino_results = dino_results_dict[key]
    kalman_results = kalman_results_dict[key]
    if dino_results['AJ'] > kalman_results['AJ']:
        total += dino_results['total']
        OA += dino_results['OA']
        AJ += dino_results['AJ']
    else:
        total += kalman_results['total']
        OA += kalman_results['OA']
        AJ += kalman_results['AJ']
total /= len(dino_results_dict.keys()) - 1
OA /= len(dino_results_dict.keys()) - 1
AJ /= len(dino_results_dict.keys()) - 1
print(f"total: {total}, OA: {OA}, AJ: {AJ}")