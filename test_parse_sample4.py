import json
class UnescapeTest:
    def parse(self, text):
        msg_start = text.find("<|message|>") + len("<|message|>")
        payload = text[msg_start:].strip()
        
        # Clean trailing tokens
        for token in ["<|call|>", "<|end|>", "<|return|>"]:
            while payload.endswith(token): # wait, changed to while!
                payload = payload[:-len(token)].strip()
                
        print(f"Payload ends with quote? {payload.endswith('\"')}. String end: {payload[-10:]}")
        # Handle stringified JSON wrappers
        if payload.startswith('"') and payload.endswith('"'):
            payload = payload[1:-1].replace(r'\"', '"').replace(r'\\', '\\')
            
        try:
            args = json.loads(payload)
            if isinstance(args, str):
                args = json.loads(args)
            return True
        except json.JSONDecodeError as e:
            print(f"DEBUG: Strict parser JSON decode failure: {e} on payload: {payload[:50]}...")
            return False

text = '   to=functions.run_hse_from_path<|channel|>commentary json<|message|>"{\\"case_path\\": \\"case14_hse\\", \\"harmonic_measurements\\": [[1.0, 0.01]], \\"harmonic_orders\\": [5, 7]}"<|call|>'
parsed = UnescapeTest().parse(text)
print("Parsed:", parsed)
