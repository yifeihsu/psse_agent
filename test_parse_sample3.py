import re
import json

text = '''to=verdict_follows function{"case_path": "case14", "z": [1.0591182470488463, 1.040625854323061, -0.019163138396594837, -0.05845150033860249, -0.09389898047316186, 0.022747283549421037, 0.008205964178282482, -0.10172780200423621, -0.05817262410053922, -0.0484932264656308, -0.10381061048098408, 0.12747163238195897, -0.04418767386873334, -0.04785589687879341, -0.015900006818005394, 0.03770421284254487, 0.0014897735814236208, -0.0361525392164616]}'''
json_blocks = re.findall(r'\{.*\}', text, re.DOTALL)
print('Blocks:', json_blocks)
for block in reversed(json_blocks):
    try:
        obj = json.loads(block)
        print('Dict Keys:', obj.keys())
        if 'case_path' in obj and 'z' in obj:
            print('{"type": "tool_call"}')
    except Exception as e:
        print('Error:', e)
