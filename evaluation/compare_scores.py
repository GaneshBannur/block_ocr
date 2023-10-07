import json, os

MANUAL_PATH = 'manual_res'
FUNCTION_PROMPT_PATH  = 'res_function_prompt'
methods = ['lcs', 'reading_order']
filename = 'avg.json'
results = {'manual': {}, 'function_prompt': {}}
tab = ' '*4

for method in methods:
    with open(os.path.join(MANUAL_PATH, method, filename), 'r') as manual_fp:
        results['manual'][method] = json.load(manual_fp)['results']
    with open(os.path.join(FUNCTION_PROMPT_PATH, method, filename), 'r') as function_prompt_fp:
        results['function_prompt'][method] = json.load(function_prompt_fp)['results']
    print(method)
    for metric_name in results['manual'][method]:
        print(tab, metric_name)
        if method=='lcs':
            print(
                f"{tab*2}"
                f"{results['manual'][method][metric_name]} -> {results['function_prompt'][method][metric_name]}"
            )
        elif method=='reading_order':
            print(
                f"{tab*2}"
                f"Max: "
                f"{results['manual'][method][metric_name]['max']}"
                f" -> "
                f"{results['function_prompt'][method][metric_name]['max']}"
            )
            print(
                f"{tab*2}"
                f"Min: "
                f"{results['manual'][method][metric_name]['min']}"
                f" -> "
                f"{results['function_prompt'][method][metric_name]['min']}"
            )
        print()
