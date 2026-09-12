"""IEEE-14 node/breaker transcription of the user-supplied 2026-09-10 figure.

Additive, versioned model: the historical pocket123 functions are NOT redefined.
65 connectivity nodes, 73 switches, 20 original electrical branches.
Buses 10/14 share a yard; 7 is a transformer star point; 8 has no drawn CB.
No tiny-impedance branches, invented switches, generator splitting, or line pruning.

Requirements: numpy; pandapower only for build_nb_ieee14_full().
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
import hashlib
import json
import numpy as np

MODEL_ID = "ieee14_full_schematic_v1"
REVIEWED_REPO_COMMIT = "a5db72c0ac6a84939b4760d8880f94262c0a4fc4"
BRANCH_PAIRS = ((1,2),(1,5),(2,3),(2,4),(2,5),(3,4),(4,5),(4,7),
                (4,9),(5,6),(6,11),(6,12),(6,13),(7,8),(7,9),(9,10),
                (9,14),(10,11),(12,13),(13,14))

@dataclass(frozen=True)
class Node:
    name: str
    planning_bus: int
    yard: str
    kind: str
    x: float
    y: float

@dataclass(frozen=True)
class Breaker:
    name: str
    a: str
    b: str
    closed: bool
    yard: str
    x: float
    y: float

@dataclass
class FullTopology:
    nodes: dict[str, Node] = field(default_factory=dict)
    breakers: list[Breaker] = field(default_factory=list)
    # Directed original (bus, neighbour) -> physical terminal node.
    terminals: dict[tuple[int, int], str] = field(default_factory=dict)
    # Fixed terminals; never split one physical unit to achieve numerical closure.
    equipment: dict[str, dict[int, str]] = field(default_factory=dict)
    anchors: dict[int, str] = field(default_factory=dict)

    def states(self, overrides: Mapping[str, bool | str | int] | None = None) -> dict[str, bool]:
        result = {c.name: c.closed for c in self.breakers}
        if overrides is not None:
            unknown = set(overrides) - result.keys()
            if unknown:
                raise ValueError(f"Unknown breaker(s): {sorted(unknown)}")
            for name, state in overrides.items():
                result[name] = parse_status(state)
        return result

    def components(self, overrides: Mapping[str, bool | str | int] | None = None) -> list[tuple[str, ...]]:
        states = self.states(overrides)
        parent = {n: n for n in self.nodes}
        def find(n: str) -> str:
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n
        for cb in self.breakers:
            if states[cb.name]:
                a, b = find(cb.a), find(cb.b)
                parent[b] = a
        groups: dict[str, list[str]] = {}
        for name in self.nodes:
            groups.setdefault(find(name), []).append(name)
        return sorted((tuple(sorted(g)) for g in groups.values()),
                      key=lambda g: (min(self.nodes[n].planning_bus for n in g), g))

    def node_to_bus(self, overrides=None) -> dict[str, int]:
        return {node: k for k, group in enumerate(self.components(overrides), start=1)
                for node in group}

    def signature(self, overrides=None, *, terminal_only: bool = False) -> tuple:
        """Label-independent closed-switch partition, not a CB-status fingerprint."""
        active = set(self.terminals.values())
        for values in self.equipment.values():
            active.update(values.values())
        groups = self.components(overrides)
        if terminal_only:
            groups = [tuple(n for n in g if n in active) for g in groups]
        return tuple(sorted(g for g in groups if g))

    def to_dict(self) -> dict:
        return {
            "model_id": MODEL_ID,
            "reviewed_repo_commit": REVIEWED_REPO_COMMIT,
            "provenance": "Manual transcription of user-supplied schematic; not an official IEEE breaker inventory.",
            "assumptions": [
                "Solid squares are closed; hollow squares are open.",
                "Buses 10 and 14 are separate normal topological buses within one shared switchyard.",
                "Bus 7 retains the case14 transformer star-equivalent; bus 8 retains its generator terminal.",
                "No breakers are invented at 7 or 8; none are visible in the supplied image.",
                "Bus-1 slack is attached to 1N1; the second arrow is not assigned an invented load.",
                "Bus-3 generator is at 3B1 and load at 3B2: the arrow types are not labelled in the image.",
                "Bus-9 shunt is attached to 9|I; no separate shunt bay is drawn.",
                "Electrical parameters and generator/load quantities come from the supplied reference case, not pixels.",
            ],
            "nodes": [asdict(n) for n in self.nodes.values()],
            "breakers": [asdict(c) for c in self.breakers],
            "branches": [{"reference_row": i, "from_bus": f, "to_bus": t,
                          "from_node": self.terminals[f,t], "to_node": self.terminals[t,f]}
                         for i, (f,t) in enumerate(BRANCH_PAIRS)],
            "equipment": self.equipment,
            "voltage_anchors": self.anchors,
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def parse_status(value: bool | str | int) -> bool:
    """Never use bool('open'), which evaluates to True."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"open", "closed"}:
        return value.strip().lower() == "closed"
    raise ValueError(f"Breaker status must be bool, 0/1, or open/closed; got {value!r}")


def build_full_topology() -> FullTopology:
    m = FullTopology()
    def n(name, bus, x, y, kind="terminal", yard=None):
        if name in m.nodes:
            raise ValueError(f"Duplicate node: {name}")
        m.nodes[name] = Node(name, bus, yard or str(bus), kind, x, y)
    def cb(name, a, b, closed, x, y, yard=None):
        m.breakers.append(Breaker(name, a, b, closed,
                          yard or m.nodes[a].yard, x, y))
    def ends(bus, mapping):
        for neighbour, node in mapping.items():
            m.terminals[bus, neighbour] = node
    def bars(bus, x, y1, y2):
        n(f"{bus}B1",bus,x,y1,"busbar")
        n(f"{bus}B2",bus,x,y2,"busbar")
    def double(bus, tag, x, y, upper=True, lower=False):
        name = f"{bus}|{tag}"
        n(name,bus,x,y)
        cb(f"CB_{bus}_{tag}_B1",name,f"{bus}B1",upper,x,(m.nodes[f"{bus}B1"].y+y)/2)
        cb(f"CB_{bus}_{tag}_B2",name,f"{bus}B2",lower,x,(m.nodes[f"{bus}B2"].y+y)/2)
        return name
    def triple(bus, k, x, y1, y2, states):
        a,b = f"{bus}N{2*k-1}", f"{bus}N{2*k}"
        n(a,bus,x,y1); n(b,bus,x,y2)
        cb(f"CB_{bus}_B1_N{2*k-1}",f"{bus}B1",a,states[0],x,(m.nodes[f"{bus}B1"].y+y1)/2)
        cb(f"CB_{bus}_N{2*k-1}_N{2*k}",a,b,states[1],x,(y1+y2)/2)
        cb(f"CB_{bus}_N{2*k}_B2",b,f"{bus}B2",states[2],x,(m.nodes[f"{bus}B2"].y+y2)/2)

    bars(1,82,310,416)
    triple(1,1,58,342,379,(True,True,True))
    triple(1,2,107,342,379,(True,False,True))
    ends(1,{5:"1N3",2:"1N4"})

    for i,x,y in [(1,106,565),(2,155,521),(3,202,521),(4,202,565),(5,155,565)]:
        n(f"2R{i}",2,x,y,"ring")
    for a,b,s,x,y in [(1,2,False,129,521),(2,3,True,178,521),(3,4,True,202,543),
                       (4,5,True,178,565),(5,1,True,129,565)]:
        cb(f"CB_2R{a}_2R{b}",f"2R{a}",f"2R{b}",s,x,y)
    ends(2,{1:"2R1",5:"2R2",4:"2R3",3:"2R4"})

    bars(3,330,529,602)
    double(3,"L32",306,565,False,True)
    double(3,"L34",354,565,True,True)
    ends(3,{2:"3|L32",4:"3|L34"})

    bars(4,445,379,486)
    for k,x,s in [(1,405,(True,True,True)),(2,445,(True,False,True)),(3,483,(True,False,True))]:
        triple(4,k,x,411,449,s)
    ends(4,{9:"4N1",5:"4N2",7:"4N3",2:"4N4",3:"4N6"})

    bars(5,256,305,379)
    for tag,x,u,l in [("L51",182,True,True),("T56",211,True,True),
                       ("L52",241,True,False),("L54",270,True,False),("I",299,True,False)]:
        double(5,tag,x,343,u,l)
    cb("CB_5_B1_B2","5B1","5B2",False,328,342)
    ends(5,{1:"5|L51",6:"5|T56",2:"5|L52",4:"5|L54"})

    bars(6,110,166,240)
    cb("CB_6_B1_B2","6B1","6B2",True,40,203)
    for tag,x,u,l in [("I",66,False,True),("L612",95,True,False),("T65",124,True,False),
                       ("L613",153,True,False),("L611",183,False,True)]:
        double(6,tag,x,203,u,l)
    ends(6,{12:"6|L612",5:"6|T65",13:"6|L613",11:"6|L611"})

    n("7STAR",7,430,349,"transformer_star")
    n("8B",8,483,350,"generator_terminal")
    ends(7,{4:"7STAR",8:"7STAR",9:"7STAR"})
    ends(8,{7:"8B"})

    bars(9,445,215,322)
    triple(9,1,405,247,286,(True,True,True))
    triple(9,2,445,247,286,(True,False,True))
    double(9,"I",483,266,False,True)
    ends(9,{10:"9N1",4:"9N2",14:"9N3",7:"9N4"})

    # A shared yard, NOT a fictional 10--14 transmission branch.
    for name,bus,x,y,kind in [("14B1",14,445,52,"busbar"),("10B1",10,445,159,"busbar"),
         ("10N1",10,405,84,"terminal"),("10N2",10,405,122,"terminal"),
         ("14N1",14,445,84,"terminal"),("14N2",14,445,122,"terminal"),
         ("14|I",14,483,84,"injection"),("10|I",10,483,122,"injection")]:
        n(name,bus,x,y,kind,"10_14")
    for name,a,b,s,x,y in [
        ("CB_Y1014_14B_10N1","14B1","10N1",False,405,67),
        ("CB_Y1014_10N1_10N2","10N1","10N2",True,405,103),
        ("CB_Y1014_10N2_10B","10N2","10B1",True,405,141),
        ("CB_Y1014_14B_14N1","14B1","14N1",True,445,67),
        ("CB_Y1014_14N1_14N2","14N1","14N2",True,445,103),
        ("CB_Y1014_14N2_10B","14N2","10B1",False,445,141),
        ("CB_Y1014_14B_I14","14B1","14|I",True,483,67),
        ("CB_Y1014_I14_I10","14|I","10|I",False,483,103),
        ("CB_Y1014_I10_10B","10|I","10B1",True,483,141)]:
        cb(name,a,b,s,x,y,"10_14")
    ends(10,{11:"10N1",9:"10N2"})
    ends(14,{13:"14N1",9:"14N2"})

    bars(11,313,47,121)
    double(11,"L116",289,85,True,True)
    double(11,"L1110",338,85,True,True)
    ends(11,{6:"11|L116",10:"11|L1110"})

    n("12B1",12,97,36,"busbar")
    n("12|L126",12,81,81)
    n("12|L1213",12,114,81)
    cb("CB_12_B1_L126","12B1","12|L126",True,81,59)
    cb("CB_12_B1_L1213","12B1","12|L1213",True,114,59)
    ends(12,{6:"12|L126",13:"12|L1213"})

    for i,x,y in [(1,151,36),(2,198,36),(3,198,81),(4,151,81)]:
        n(f"13R{i}",13,x,y,"ring")
    for a,b,s,x,y in [(1,2,True,174,36),(2,3,False,198,59),
                       (3,4,True,174,81),(4,1,True,151,59)]:
        cb(f"CB_13R{a}_13R{b}",f"13R{a}",f"13R{b}",s,x,y)
    ends(13,{14:"13R2",6:"13R3",12:"13R4"})

    anchors={1:"1N1",2:"2R5",3:"3B1",4:"4N5",5:"5|I",6:"6|I",7:"7STAR",
             8:"8B",9:"9|I",10:"10|I",11:"11B2",12:"12B1",13:"13R1",14:"14|I"}
    load=dict(anchors); load[3]="3B2"
    m.anchors=anchors
    m.equipment={"gen":dict(anchors),"ext_grid":dict(anchors),"load":load,
                 "sgen":dict(anchors),"shunt":dict(load)}
    validate_model(m)
    return m


def validate_model(m: FullTopology) -> None:
    names=[c.name for c in m.breakers]
    if len(names)!=len(set(names)):
        raise ValueError("Duplicate breaker identifiers")
    for cb in m.breakers:
        if cb.a not in m.nodes or cb.b not in m.nodes or cb.a==cb.b:
            raise ValueError(f"Invalid breaker endpoints: {cb}")
    expected={(f,t) for f,t in BRANCH_PAIRS}|{(t,f) for f,t in BRANCH_PAIRS}
    if set(m.terminals)!=expected:
        raise ValueError("Expected exactly 40 IEEE-14 branch ends")
    mapping=m.node_to_bus()
    if len(m.components())!=14:
        raise ValueError("Normal state does not give 14 topological buses")
    for name,node in m.nodes.items():
        if mapping[name]!=node.planning_bus:
            raise ValueError(f"Normal component of {name} disagrees with its planning bus")
    for (bus,_),node in m.terminals.items():
        if node not in m.nodes or mapping[node]!=bus:
            raise ValueError(f"Wrong physical terminal {bus}: {node}")
    for table,attach in m.equipment.items():
        if set(attach)!=set(range(1,15)):
            raise ValueError(f"Incomplete equipment attachment policy for {table}")
        for bus,node in attach.items():
            if node not in m.nodes or mapping[node]!=bus:
                raise ValueError(f"Invalid {table} attachment {bus}: {node}")


def single_flip_audit(model: FullTopology | None = None) -> list[dict]:
    m=model or build_full_topology()
    baseline=m.signature()
    terminal_baseline=m.signature(terminal_only=True)
    result=[]
    for cb in m.breakers:
        state={cb.name:not cb.closed}
        ncomp=len(m.components(state))
        result.append({"cb_name":cb.name,"yard":cb.yard,"normal_closed":cb.closed,
            "flipped_closed":not cb.closed,"topological_buses":ncomp,
            "partition_changed":m.signature(state)!=baseline,
            "terminal_partition_changed":m.signature(state,terminal_only=True)!=terminal_baseline,
            "effect":"split" if ncomp>14 else "merge" if ncomp<14 else "equivalent",
            "ground_truth_status":not cb.closed,"reported_status":cb.closed})
    return result


def _validate_case(case: Mapping[str, Any]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    bus=np.asarray(case["bus"],dtype=float)
    gen=np.asarray(case["gen"],dtype=float)
    branch=np.asarray(case["branch"],dtype=float)
    if bus.ndim!=2 or bus.shape[0]!=14 or bus.shape[1]<13:
        raise ValueError("Reference must contain 14 MATPOWER-format buses")
    if list(bus[:,0].astype(int))!=list(range(1,15)):
        raise ValueError("Reference bus rows must be ordered 1..14; remap explicitly first")
    expected={tuple(sorted(p)) for p in BRANCH_PAIRS}
    actual=[tuple(sorted(map(int,r[:2]))) for r in branch]
    if len(actual)!=20 or len(set(actual))!=20 or set(actual)!=expected:
        raise ValueError("Reference branches must match the 20 unique IEEE-14 bus pairs")
    if not np.all(np.isfinite(bus[:,:13])) or not np.all(np.isfinite(branch[:,:13])):
        raise ValueError("Nonfinite electrical case values")
    if gen.ndim!=2 or gen.shape[1]<10 or not set(gen[:,0].astype(int))<=set(range(1,15)):
        raise ValueError("Invalid generator matrix")
    return bus,gen,branch


def topology_to_matpower(reference_case: Mapping[str, Any], status_map=None,
                         *, model: FullTopology | None = None) -> tuple[dict,dict]:
    """Contract ideal closed switches, retaining ALL finite-impedance branches.

    Branch/gen row identities and electrical/control data are preserved. An open
    branch-end breaker produces a dangling terminal, not an out-of-service line:
    line charging is retained. Empty isolated busbars are type 4, not deleted.
    No loads are silently shed and no unsupplied island is silently removed.
    """
    m=model or build_full_topology()
    validate_model(m)
    src_bus,src_gen,src_branch=_validate_case(reference_case)
    groups=m.components(status_map); lookup=m.node_to_bus(status_map)
    bus=np.zeros((len(groups),src_bus.shape[1]),dtype=float)
    for i,group in enumerate(groups):
        representative=min(m.nodes[n].planning_bus for n in group)
        bus[i]=src_bus[representative-1]
        bus[i,0]=i+1; bus[i,1]=1; bus[i,2:6]=0
    for row in src_bus:
        b=int(row[0])
        load=lookup[m.equipment["load"][b]]-1
        shunt=lookup[m.equipment["shunt"][b]]-1
        bus[load,2:4]+=row[2:4]
        bus[shunt,4:6]+=row[4:6]
    gen=src_gen.copy()
    controlled={}
    for i,row in enumerate(src_gen):
        b=int(row[0]); target=lookup[m.equipment["gen"][b]]
        gen[i,0]=target
        if row[7]>0:
            if target in controlled and not np.isclose(controlled[target],row[5],rtol=0,atol=1e-9):
                raise ValueError(f"Conflicting voltage controllers fused at bus {target}")
            controlled[target]=row[5]
            bus[target-1,1]=max(bus[target-1,1],3 if src_bus[b-1,1]==3 else 2)
            bus[target-1,7]=row[5]
    branch=src_branch.copy()
    for i,row in enumerate(src_branch):
        f,t=map(int,row[:2])
        branch[i,0]=lookup[m.terminals[f,t]]
        branch[i,1]=lookup[m.terminals[t,f]]
    degree=np.zeros(len(bus),dtype=int)
    for row in branch:
        if row[10]>0:
            degree[int(row[0])-1]+=1; degree[int(row[1])-1]+=1
    for i,row in enumerate(bus):
        if degree[i]==0 and row[1]==1 and np.all(row[2:6]==0):
            bus[i,1]=4
    result=deepcopy(dict(reference_case))
    result.update(bus=bus,gen=gen,branch=branch)
    result.pop("order",None); result.pop("success",None)
    if "bus_name" in result:
        result["bus_name"]=[" / ".join(g) for g in groups]
    info={"model_id":MODEL_ID,"model_fingerprint":m.fingerprint(),
          "node_to_bus":lookup,"components":[list(g) for g in groups],
          "status_map":m.states(status_map),"branch_reference_rows":list(range(len(branch))),
          "inactive_empty_busbars":bus[bus[:,1]==4,0].astype(int).tolist(),
          "load_p_mw":float(bus[:,2].sum()),"load_q_mvar":float(bus[:,3].sum())}
    return result,info


def write_matpower_case(case: Mapping[str, Any], filename, name="case14_full_topology") -> None:
    """Write text .m matrices, compatible with the repository's text case parser."""
    from pathlib import Path
    import re
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*",name):
        raise ValueError("Invalid MATLAB function name")
    chunks=[f"function mpc = {name}",f"% {MODEL_ID}; ideal-switch topology processed",
            "mpc.version = '2';",f"mpc.baseMVA = {float(case['baseMVA']):.17g};"]
    for key in ("bus","gen","branch","gencost"):
        if key in case:
            a=np.asarray(case[key],dtype=float)
            chunks.append(f"mpc.{key} = [")
            chunks.extend("  "+" ".join(f"{v:.17g}" for v in row)+";" for row in a)
            chunks.append("];")
    path=Path(filename); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(chunks)+"\n",encoding="utf-8")


def build_nb_ieee14_full(status_map=None, *, reference_net=None, model=None):
    """Build pandapower NB net with ideal bus-bus switches (not impedances).

    Returns (net, sec_bus, cb_idx, line_idx, trafo_idx), but cb_idx indexes
    net.switch, NOT net.impedance. Use pp.runpp(); the legacy pocket CB-flow
    extractor is intentionally not used. The electrical element tables are
    copied wholesale; all tap settings, Q limits, status, and costs are retained.
    Optional dependency; this adapter needs validation in the target pp version.
    """
    import pandas as pd
    import pandapower as pp
    import pandapower.networks as pn
    m=model or build_full_topology(); validate_model(m)
    states=m.states(status_map)
    ref=deepcopy(reference_net if reference_net is not None else pn.case14())
    if len(ref.bus)!=14:
        raise ValueError("Expected the 14-bus reference network")
    for table in ("switch","impedance","trafo3w","dcline"):
        if table in ref and len(ref[table]):
            raise ValueError(f"Reference {table} table is not supported by this IEEE-14 adapter")
    # In pn.case14(), the sorted original indices correspond to planning buses 1..14.
    original=sorted(ref.bus.index)
    index_to_plan={idx:i+1 for i,idx in enumerate(original)}
    observed=[]
    for table,cols in (("line",("from_bus","to_bus")),("trafo",("hv_bus","lv_bus"))):
        for _,r in ref[table].iterrows():
            observed.append(tuple(sorted((index_to_plan[int(r[cols[0]])],index_to_plan[int(r[cols[1]])]))))
    if len(observed)!=20 or set(observed)!={tuple(sorted(p)) for p in BRANCH_PAIRS}:
        raise ValueError("Reference bus indexing / physical branch mapping does not match IEEE-14")
    net=deepcopy(ref)
    sec_bus={name:i for i,name in enumerate(m.nodes)}
    rows=[]
    for name,node in m.nodes.items():
        row=ref.bus.loc[original[node.planning_bus-1]].copy()
        row["name"]=name
        row["planning_bus"]=node.planning_bus
        row["yard"]=node.yard
        rows.append(row)
    net.bus=pd.DataFrame(rows,index=range(len(rows)))
    for table in ("ext_grid","gen","load","sgen","shunt"):
        if table in net and len(net[table]):
            net[table]["bus"]=[sec_bus[m.equipment[table][index_to_plan[int(b)]]]
                               for b in ref[table]["bus"]]
    line_idx={}; trafo_idx={}
    for table,cols,prefix,out in (("line",("from_bus","to_bus"),"line",line_idx),
                                  ("trafo",("hv_bus","lv_bus"),"trafo",trafo_idx)):
        for i,r in ref[table].iterrows():
            f,t=(index_to_plan[int(r[c])] for c in cols)
            net[table].at[i,cols[0]]=sec_bus[m.terminals[f,t]]
            net[table].at[i,cols[1]]=sec_bus[m.terminals[t,f]]
            name=f"{prefix}_{f}-{t}"
            net[table].at[i,"name"]=name; out[name]=int(i)
    cb_idx={}
    for c in m.breakers:
        idx=pp.create_switch(net,bus=sec_bus[c.a],element=sec_bus[c.b],et="b",
                             closed=states[c.name],type="CB",name=c.name,z_ohm=0)
        cb_idx[c.name]=idx
    for key in list(net):
        if key.startswith("res_") and isinstance(net[key],pd.DataFrame):
            net[key]=net[key].iloc[0:0].copy()
    net["_ppc"]=None; net["converged"]=False
    net["topology_model_id"]=MODEL_ID
    net["topology_model_fingerprint"]=m.fingerprint()
    return net,sec_bus,cb_idx,line_idx,trafo_idx
