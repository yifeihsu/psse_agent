import json

# Simulated string from the model (with literal backslashes escaping quotes inside)
block = r'{\"case_path\": \"case14_topo_balanced\"}'
print(f"Original length: {len(block)}")
print(f"Original string: {block}")

if r'\"' in block:
    print("Found backslash-quote!")
    clean = block.replace(r'\"', '"').replace(r'\\', '\\')
    print(f"Clean string: {clean}")
    
    try:
        obj = json.loads(clean)
        print("Success:", obj)
    except Exception as e:
        print("Decode Error:", e)
