import requests
import json
import numpy as np

# We'll use the wls_from_path tool endpoint.
# Build a dummy measurement vector matching case14 sizes:
# nb=14, nl=20 => length = 3*14 + 4*20 = 42 + 80 = 122
z = np.ones(122).tolist()

payload = {
    "case_path": "case14.m",
    "z": z
}

try:
    response = requests.post("http://127.0.0.1:3929/mcp/tools/wls_from_path", json=payload)
    print("wls_from_path Status:", response.status_code)
    try:
        data = response.json()
        print("Success:", data.get("success", False))
        if "lambdaN" in data:
            print("lambdaN length:", len(data["lambdaN"]))
            print("r length:", len(data["r"]))
        else:
            print("Response:", data)
    except json.JSONDecodeError as e:
         print("JSON Error:", e, "| Response Text:", response.text)

except Exception as e:
    print("Request failed:", e)
