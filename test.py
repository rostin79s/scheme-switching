from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

# Operations and their types
operations = ['op2', 'op3', 'op1', 'op4']
operation_types = {'op1': 'cmp', 'op2': 'mul', 'op3': 'mul', 'op4': 'mul'}
# Execution times
execution_times = {
    'TFHE': {'cmp': 1, 'mul': 11, 'add': 2},
    'CKKS': {'cmp': 100, 'mul': 1, 'add': 1}
}

# Scheme switching costs
switch_cost_CKKS_to_TFHE = 5   # Cost of switching from CKKS to TFHE
switch_cost_TFHE_to_CKKS = 10  # Cost of switching from TFHE to CKKS

# Create the ILP problem
prob = LpProblem("FHE_Scheme_Selection", LpMinimize)

# Decision variables for schemes
scheme = {op: {s: LpVariable(f"x_{op}_{s}", 0, 1, cat="Binary") for s in ['TFHE', 'CKKS']} for op in operations}

# Decision variables for switching types
switch_CKKS_to_TFHE = {i: LpVariable(f"switch_CKKS_to_TFHE_{i}", 0, 1, cat="Binary") for i in range(len(operations) - 1)}
switch_TFHE_to_CKKS = {i: LpVariable(f"switch_TFHE_to_CKKS_{i}", 0, 1, cat="Binary") for i in range(len(operations) - 1)}

# Objective function: Operation costs + switching costs
operation_cost = lpSum(
    scheme[op]['TFHE'] * execution_times['TFHE'][operation_types[op]] +
    scheme[op]['CKKS'] * execution_times['CKKS'][operation_types[op]]
    for op in operations
)

switch_cost_total = lpSum(
    switch_CKKS_to_TFHE[i] * switch_cost_CKKS_to_TFHE + 
    switch_TFHE_to_CKKS[i] * switch_cost_TFHE_to_CKKS
    for i in range(len(operations) - 1)
)

prob += operation_cost + switch_cost_total

# Constraint: Each operation is assigned to exactly one scheme
for op in operations:
    prob += scheme[op]['TFHE'] + scheme[op]['CKKS'] == 1

# Switching logic constraints
for i in range(len(operations) - 1):
    op1, op2 = operations[i], operations[i + 1]
    
    # Switch from CKKS to TFHE
    prob += switch_CKKS_to_TFHE[i] >= scheme[op1]['CKKS'] + scheme[op2]['TFHE'] - 1
    
    # Switch from TFHE to CKKS
    prob += switch_TFHE_to_CKKS[i] >= scheme[op1]['TFHE'] + scheme[op2]['CKKS'] - 1

# Solve the problem
prob.solve()

# Output results
print("Optimal Operation Assignments:")
for op in operations:
    assigned_scheme = 'TFHE' if value(scheme[op]['TFHE']) == 1 else 'CKKS'
    print(f"  {op}: {assigned_scheme}")

print("\nSwitching Decisions:")
for i in range(len(operations) - 1):
    if value(switch_CKKS_to_TFHE[i]) == 1:
        print(f"  Switch from CKKS to TFHE between {operations[i]} and {operations[i + 1]}")
    elif value(switch_TFHE_to_CKKS[i]) == 1:
        print(f"  Switch from TFHE to CKKS between {operations[i]} and {operations[i + 1]}")
