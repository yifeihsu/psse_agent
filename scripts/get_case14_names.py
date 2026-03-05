import pandapower.networks as pn
import pandapower as pp
from pypower.api import case14

def get_case14_names():
    # 1. Get Standard Order from pypower
    ppc = case14()
    branch = ppc['branch']
    
    # 2. Get Pandapower Case14 to get names
    net = pn.case14()
    
    # Map (f, t) -> Name
    # Note: pypower uses 1-based indexing. pandapower uses 0-based.
    # We need to match (f, t) pairs.
    
    # Build lookup from net.line
    line_map = {}
    for idx in net.line.index:
        fb = net.line.at[idx, 'from_bus']
        tb = net.line.at[idx, 'to_bus']
        # nodebreaker_pp14 naming convention:
        # name=f"line_{fb}-{tb}" (using 1-based or 0-based? nodebreaker uses pp2num map)
        # nodebreaker_pp14 uses:
        # fb = pp2num[fb_pp]; tb = pp2num[tb_pp]
        # name=f"line_{fb}-{tb}"
        # pp2num maps 0-based index to 0-based index if names are missing.
        # But case14 has bus names?
        # Let's check nodebreaker logic carefully.
        # It maps "planning bus number" to pp index.
        # If case14 has no names, it uses i+1.
        
        # Let's assume nodebreaker uses 1-based naming if original was 1-based.
        # Actually, let's just use the (f, t) from pypower and construct "line_f-t" or "trafo_f-t".
        
        # We need to know if it's a line or trafo.
        # In pypower, lines have tap=0 (usually). Trafos have tap != 0.
        # Column 8 (index 8) is ratio. 0 means line (or ratio 1? no, 0 means line).
        pass

    print("Standard Case14 Branch Order (Name Construction):")
    names = []
    for i in range(branch.shape[0]):
        f = int(branch[i, 0])
        t = int(branch[i, 1])
        ratio = branch[i, 8]
        
        # In pypower case14, f and t are 1-based.
        # nodebreaker_pp14 uses 1-based naming for case14 because it maps bus indices.
        # Let's assume names are "line_{f}-{t}" or "trafo_{f}-{t}".
        
        if ratio == 0.0:
            name = f"line_{f}-{t}"
        else:
            name = f"trafo_{f}-{t}"
        
        names.append(name)
        print(f"Index {i}: {name}")

    print("\nCOPY THIS LIST:")
    print(names)

if __name__ == "__main__":
    get_case14_names()
